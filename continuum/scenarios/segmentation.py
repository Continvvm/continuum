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

    Scenario: Each new tasks bring new classes only

    :param cl_dataset: A continual dataset.
    :param nb_tasks: The scenario number of tasks.
    :param increment: Either number of classes per task (e.g. increment=2),
                    or a list specifying for every task the amount of new classes
                     (e.g. increment=[5,1,1,1,1]).
    :param initial_increment: A different task size applied only for the first task.
                              Desactivated if `increment` is a list.
    :param transformations: A list of transformations applied to all tasks.
    :param class_order: An optional custom class order, used for NC.
                        e.g. [0,1,2,3,4,5,6,7,8,9] or [5,2,4,1,8,6,7,9,0,3]
    """

    def __init__(
        self,
        cl_dataset: _ContinuumDataset,
        nb_classes: int = 0,
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
        return len(np.unique(self.dataset[1]))  # type: ignore

    def __getitem__(self, task_index: Union[int, slice]):
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

        if self.mode in ("overlap", "disjoint"):
            # Previous and future (for disjoint) classes are hidden
            labels = self._get_task_labels(task_index)
        elif self.mode == "sequential":
            # Previous classes are not hidden, no future classes are present
            if isinstance(task_index, int):
                labels = self._get_task_labels(list(range(task_index + 1)))
            else:
                labels = self._get_task_labels(list(range(max(task_index) + 1)))
        else:
            raise ValueError(f"Unknown mode={mode}.")

        inverted_order = {label: self.class_order.index(label) + 1 for label in labels}
        inverted_order[255] = 255

        masking_value = 0
        if not self.cl_dataset.train:
            if self.test_background:
                inverted_order[0] = 0
            else:
                masking_value = 255

        label_trsf = torchvision.transforms.Lambda(
            lambda seg_map: seg_map.apply_(
                lambda v: inverted_order.get(v, masking_value)
            )
        )

        return TaskSet(x, y, t, self.trsf, target_trsf=label_trsf, data_type=self.cl_dataset.data_type)

    def get_original_targets(self, targets: np.ndarray) -> np.ndarray:
        """Returns the original targets not changed by the custom class order.

        :param targets: An array of targets, as provided by the task datasets.
        :return: An array of targets, with their original values.
        """
        return self._class_mapping(targets)

    def _get_task_ids(self, t, task_indexes):
        if isinstance(task_indexes, list):
            task_indexes = max(task_indexes)
        return np.ones((len(t))) * task_indexes

    def _get_task_labels(self, task_indexes: Union[int, List[int]]) -> List[int]:
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


def _filter_images(paths, increments, class_order, mode="overlap"):
    """Select images corresponding to the labels.

    Strongly inspired from Cermelli's code:
    https://github.com/fcdl94/MiB/blob/master/dataset/utils.py#L19
    """
    indexes_to_classes = []
    pb = ProgressBar()

    with multiprocessing.Pool(min(8, multiprocessing.cpu_count())) as pool:
        for i, classes in enumerate(pool.imap(_find_classes, paths)):
            indexes_to_classes.append(classes)
            if i % 100 == 0:
                pb.update(None, i, len(paths))

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


def _find_classes(path):
    return np.unique(np.array(Image.open(path)).reshape(-1))
