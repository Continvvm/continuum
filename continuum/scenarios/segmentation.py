import warnings
from copy import copy
from typing import Callable, List, Union, Optional
import os
import multiprocessing

import numpy as np
from PIL import Image
import torchvision

from continuum.datasets import _ContinuumDataset
from continuum.scenarios import ClassIncremental
from continuum.tasks import TaskSet
from continuum.download import ProgressBar


class SegmentationClassIncremental(ClassIncremental):
    """Continual Loader, generating datasets for the consecutive tasks.

    References
        * Incremental Learning Techniques for Semantic Segmentation
          Umberto Michieli, Pietro Zanuttigh
          ICCV Workshop 2017
        * Modeling the Background for Incremental Learning in Semantic Segmentation
          Fabio Cermelli, Massimiliano Mancini, Samuel Rota Bulò, Elisa Ricci, Barbara Caputo
          CVPR 2020
        * PLOP: Learning without Forgetting for Continual Semantic Segmentation
          Arthur Douillard, Yifu Chen, Arnaud Dapogny, Matthieu Cord
          CVPR 2021

    :param cl_dataset: A continual dataset.
    :param nb_classes: The number of classes of the dataset (excluding bg=0 and unk=255).
    :param increment: Either number of classes per task (e.g. increment=2),
                    or a list specifying for every task the amount of new classes
                     (e.g. increment=[5,1,1,1,1]).
    :param initial_increment: A different task size applied only for the first task.
                              Desactivated if `increment` is a list.
    :param transformations: A list of transformations applied to all tasks.
    :param class_order: An optional custom class order, used for NC.
                        e.g. [0,1,2,3,4,5,6,7,8,9] or [5,2,4,1,8,6,7,9,0,3]
    :param mode: The mode of incremental segmentation. In both "sequential" and
                 "disjoint", images only contain pixels of old or current classes,
                 while in "overlap", future classes can also be present.
                 In "sequential" all pixels are labelized while in "disjoint" and
                 "overlap" only pixels of the current classes are labelized.
    :param save_indexes: Path where to save and load the indexes of the different
                         tasks. Computing it may be slow, so it can be worth to
                         checkpoint those.
    :param test_background: Whether to ignore the background (0) during the testing
                            phase (False) or to keep its label (True).
    """

    def __init__(
        self,
        cl_dataset: _ContinuumDataset,
        nb_classes: int,
        increment: Union[List[int], int] = 0,
        initial_increment: int = 0,
        transformations: List[Callable] = None,
        class_order: Optional[List[int]] = None,
        mode: str = "overlap",
        save_indexes: Optional[str] = None,
        test_background: bool = True
    ) -> None:
        self.mode = mode
        self.save_indexes = save_indexes
        self.test_background = test_background
        self._nb_classes = nb_classes

        if cl_dataset.data_type != "segmentation":
            raise ValueError(
                f"Dataset {cl_dataset} doesn't have the right data type but "
                f"{self.cl_dataset.data_type}."
            )

        if self.mode not in ("overlap", "disjoint", "sequential"):
            raise ValueError(f"Unknown mode={mode}.")

        if class_order is not None:
            if 0 in class_order:
                raise ValueError("Exclude the background (0) from the class order.")
            if 255 in class_order:
                raise ValueError("Exclude the unknown (255) from the class order.")
            if len(class_order) != nb_classes:
                raise ValueError(
                    f"Number of classes ({nb_classes}) != class ordering size ({len(class_order)}."
                )

        super().__init__(
            cl_dataset=cl_dataset,
            increment=increment,
            initial_increment=initial_increment,
            class_order=class_order,
            transformations=transformations,
        )

    @property
    def nb_classes(self) -> int:
        """Total number of classes in the whole continual setting."""
        return self._nb_classes

    def __getitem__(self, task_index: Union[int, slice]) -> TaskSet:
        """Returns a task by its unique index.

        :param task_index: The unique index of a task. As for List, you can use
                           indexing between [0, len], negative indexing, or
                           even slices.
        :return: A train PyTorch's Datasets.
        """
        if isinstance(task_index, slice) and task_index.step is not None:
            raise ValueError("Step in slice for segmentation is not supported.")

        x, y, t, task_index = self._select_data_by_task(task_index)
        t = self._get_task_ids(t, task_index)

        return TaskSet(
            x, y, t,
            self.trsf,
            target_trsf=self._get_label_transformation(task_index),
            data_type=self.cl_dataset.data_type
        )

    def get_original_targets(self, targets: np.ndarray) -> np.ndarray:
        """Returns the original targets not changed by the custom class order.

        :param targets: An array of targets, as provided by the task datasets.
        :return: An array of targets, with their original values.
        """
        return self._class_mapping(targets)

    def _get_label_transformation(self, task_index: Union[int, List[int]]):
        """Returns the transformation to apply on the GT segmentation maps.

        :param task_index: The selected task id.
        :return: A pytorch transformation.
        """
        if isinstance(task_index, int):
            task_index = [task_index]
        if not self.train:
            # In testing mode, all labels brought by previous tasks are revealed
            task_index = list(range(max(task_index) + 1))

        if self.mode in ("overlap", "disjoint"):
            # Previous and future (for disjoint) classes are hidden
            labels = self._get_task_labels(task_index)
        elif self.mode == "sequential":
            # Previous classes are not hidden, no future classes are present
            labels = self._get_task_labels(list(range(max(task_index) + 1)))
        else:
            raise ValueError(f"Unknown mode={mode}.")

        inverted_order = {label: self.class_order.index(label) + 1 for label in labels}
        inverted_order[255] = 255

        masking_value = 0
        if not self.train:
            if self.test_background:
                inverted_order[0] = 0
            else:
                masking_value = 255

        return torchvision.transforms.Lambda(
            lambda seg_map: seg_map.apply_(
                lambda v: inverted_order.get(v, masking_value)
            )
        )

    def _get_task_ids(self, t: np.ndarray, task_indexes: Union[int, List[int]]) -> np.ndarray:
        """Reduce multiple task ids to a single one per sample.

        In segmentation, the same image can have several task ids. We assume that
        the selected task ids is the last one as it will be the one that matter
        for the ground-truth segmentation maps.

        :param t: A matrix of task ids of shape (nb_samples, nb_tasks).
        :param task_indexes: The selected task ids.
        :return: An array of task ids of shape (nb_samples,).
        """
        if isinstance(task_indexes, list):
            task_indexes = max(task_indexes)
        return np.ones((len(t))) * task_indexes

    def _get_task_labels(self, task_indexes: Union[int, List[int]]) -> List[int]:
        """Returns the labels/classes of the current tasks, not all present labels!

        :param task_indexes: The selected task ids.
        :return: A list of class/labels ids.
        """
        if isinstance(task_indexes, int):
            task_indexes = [task_indexes]

        labels = set()
        for t in task_indexes:
           previous_inc = sum(self._increments[:t])
           labels.update(
               self.class_order[previous_inc:previous_inc+self._increments[t]]
           )

        return list(labels)

    def _setup(self, nb_tasks: int) -> int:
        """Setups the different tasks."""
        x, y, _ = self.cl_dataset.get_data()
        self.class_order = self.class_order or self.cl_dataset.class_order or list(
            range(1, self._nb_classes + 1))

        # For when the class ordering is changed,
        # so we can quickly find the original labels
        def class_mapping(c):
            if c in (0, 255): return c
            return self.class_order[c - 1]
        self._class_mapping = np.vectorize(class_mapping)

        self._increments = self._define_increments(
            self.increment, self.initial_increment, self.class_order
        )

        # Checkpointing the indexes if the option is enabled.
        # The filtering can take multiple minutes, thus saving/loading them can
        # be useful.
        if self.save_indexes is not None and os.path.exists(self.save_indexes):
            print(f"Loading previously saved indexes ({self.save_indexes}).")
            t = np.load(self.save_indexes)
        else:
            print("Computing indexes, it may be slow!")
            t = _filter_images(
                y, self._increments, self.class_order, self.mode
            )
            if self.save_indexes is not None:
                np.save(self.save_indexes, t)

        assert len(x) == len(y) == len(t) and len(t) > 0

        self.dataset = (x, y, t)

        return len(self._increments)


def _filter_images(
    paths: Union[np.ndarray, List[str]],
    increments: List[int],
    class_order: List[int],
    mode: str = "overlap"
) -> np.ndarray:
    """Select images corresponding to the labels.

    Strongly inspired from Cermelli's code:
    https://github.com/fcdl94/MiB/blob/master/dataset/utils.py#L19

    :param paths: An iterable of paths to gt maps.
    :param increments: All individual increments.
    :param class_order: The class ordering, which may not be [1, 2, ...]. The
                        background class (0) and unknown class (255) aren't
                        in this class order.
    :param mode: Mode of the segmentation (see scenario doc).
    :return: A binary matrix representing the task ids of shape (nb_samples, nb_tasks).
    """
    indexes_to_classes = []
    pb = ProgressBar()

    with multiprocessing.Pool(min(8, multiprocessing.cpu_count())) as pool:
        for i, classes in enumerate(pool.imap(_find_classes, paths), start=1):
            indexes_to_classes.append(classes)
            if i % 100 == 0:
                pb.update(None, 100, len(paths))
        pb.end(len(paths))

    t = np.zeros((len(paths), len(increments)))
    accumulated_inc = 0

    for task_id, inc in enumerate(increments):
        labels = class_order[accumulated_inc:accumulated_inc+inc]
        old_labels = class_order[:accumulated_inc]
        all_labels = labels + old_labels + [0, 255]

        for index, classes in enumerate(indexes_to_classes):
            if mode == "overlap":
                if any(c in labels for c in classes):
                    t[index, task_id] = 1
            elif mode in ("disjoint", "sequential"):
                if any(c in labels for c in classes) and all(c in all_labels for c in classes):
                    t[index, task_id] = 1
            else:
                raise ValueError(f"Unknown mode={mode}.")

        accumulated_inc += inc

    return t


def _find_classes(path: str) -> np.ndarray:
    """Open a ground-truth segmentation map image and returns all unique classes
    contained.

    :param path: Path to the image.
    :return: Unique classes.
    """
    return np.unique(np.array(Image.open(path)).reshape(-1))
