from typing import Callable, List, Tuple, Union
from copy import copy

import numpy as np
import torch

from clloader.datasets import BaseDataset
from clloader import TaskSet
from torch.utils.data import Dataset as TorchDataset
from torchvision import transforms

class CLLoader:
    """Continual Loader, generating datasets for the consecutive tasks.

    :param cl_dataset: A continual dataset.
    :param increment: Either number of classes per task, or a list specifying for
                      every task the amount of new classes.
    :param initial_increment: A different task size applied only for the first task.
                              Desactivated if `increment` is a list.
    :param train_transformations: A list of data augmentation applied to the train set.
    :param common_transformations: A list of transformations applied to both the
                                   the train set and the test set. i.e. normalization,
                                   resizing, etc.
    :param evaluate_on: How to evaluate on val/test, either on all `seen` classes,
                        on the `current` classes, or on `all` classes.
    :param class_order: An optional custom class order, used for NC.
    """

    def __init__(
        self,
        cl_dataset: BaseDataset,
        increment: Union[List[int], int],
        initial_increment: int = 0,
        train_transformations: List[Callable] = None,
        common_transformations: List[Callable] = None,
        evaluate_on="seen",
        class_order=None
    ) -> None:

        self.cl_dataset = cl_dataset

        if train_transformations is None:
            train_transformations = []
        if common_transformations is None:
            common_transformations = self.cl_dataset.transformations
        self.train_trsf = transforms.Compose(train_transformations + common_transformations)
        self.test_trsf = transforms.Compose(common_transformations)

        if evaluate_on not in ("seen", "current", "all"):
            raise NotImplementedError(f"Evaluate mode {evaluate_on} is not supported.")
        self.evaluate_on = evaluate_on

        self._setup(increment, initial_increment, class_order)

    def _setup(self,increment: Union[List[int], int],
               initial_increment: int,
               class_order: Union[None, List[int]] = None) -> None:

        (train_x, train_y), (test_x, test_y) = self.cl_dataset.init()
        unique_classes = np.unique(train_y)

        self.class_order = class_order or self.cl_dataset.class_order or list(
            range(len(unique_classes))
        )

        self.increments = self._define_increments(increment, initial_increment)

        if len(np.unique(self.class_order)) != len(self.class_order):
            raise ValueError(f"Invalid class order, duplicates found: {self.class_order}.")

        mapper = np.vectorize(lambda x: self.class_order.index(x))
        train_y = mapper(train_y)
        test_y = mapper(test_y)

        train_t = self._set_task_labels(train_y, self.increments)
        test_t = self._set_task_labels(test_y, self.increments)

        self.train_data = (train_x, train_y, train_t)  # (data, class label, task label)
        self.test_data = (test_x, test_y, test_t)  # (data, class label, task label)
        self.class_order = np.array(self.class_order)

    def _set_task_labels(self, y, increments):

        t = copy(y) # task label as same size as y
        for task_index, increment in enumerate(increments):
            max_class = sum(self.increments[:task_index + 1])
            min_class = sum(self.increments[:task_index])  # 0 when task_index == 0.

            indexes = np.where(np.logical_and(y >= min_class, y < max_class))[0]
            t[indexes] = task_index
        return t

    def _define_increments(self, increment: Union[List[int], int],
                           initial_increment: int) -> List[int]:
        if isinstance(increment, list):

            # Check if the total number of classes is compatible between increment list and self.nb_classes
            if not sum(increment)== self.nb_classes():
                raise Exception(
                    "The increment list is not compatible with the number of classes"
                )

            increments=increment
        else:
            increments = []
            if initial_increment:
                increments.append(initial_increment)

            nb_tasks = (self.nb_classes - initial_increment) / increment
            if not nb_tasks.is_integer():
                raise Exception(
                    "The tasks won't have an equal number of classes"
                    f" with {len(self.class_order)} and increment {increment}"
                )
            increments.extend([increment for _ in range(int(nb_tasks))])

        return increments





    def get_original_targets(self, targets: np.ndarray) -> np.ndarray:
        """Returns the original targets not changed by the custom class order.

        :param targets: An array of targets, as provided by the task datasets.
        :return: An array of targets, with their original values.
        """
        return self.class_order[targets]

    @property
    def nb_classes(self) -> int:
        """Total number of classes in the whole continual setting."""
        return len(np.unique(self.train_data[1]))

    @property
    def nb_tasks(self) -> int:
        """Number of tasks in the whole continual setting."""
        return len(self)

    def __len__(self) -> int:
        """Returns the number of tasks.

        :return: Number of tasks.
        """
        return len(self.increments)

    def __iter__(self):
        """Used for iterating through all tasks with the CLLoader in a for loop."""
        self._counter = 0
        return self

    def __next__(self) -> Tuple[TaskSet, TaskSet]:
        """An iteration/task in the for loop."""
        if self._counter >= len(self):
            raise StopIteration
        task = self[self._counter]
        self._counter += 1
        return task

    def __getitem__(self, task_index):
        """Returns a task by its unique index.

        :param task_index: The unique index of a task, between 0 and len(loader) - 1.
        :return: A train and test PyTorch's Datasets.
        """
        max_class = sum(self.increments[:task_index + 1])
        min_class = sum(self.increments[:task_index])  # 0 when task_index == 0.

        train = self._select_data_by_task(task_index, split="train")
        train_dataset = TaskSet(*train, self.train_trsf, open_image=not self.cl_dataset.in_memory)

        # TODO: validation
        if self.evaluate_on == "seen":
            test = self._select_data_by_classes(0, max_class, split="test")
        elif self.evaluate_on == "current":
            test = self._select_data_by_classes(min_class, max_class, split="test")
        else:  # all
            test = self._select_data_by_classes(0, self.nb_classes, split="test")

        test_dataset = TaskSet(*test, self.test_trsf, open_image=not self.cl_dataset.in_memory)

        return train_dataset, test_dataset


    def _select_data_by_task(self,ind_task: int, split: str="train"):
        """Selects a subset of the whole data for a given task.

        :param ind_task: task index
        :param split: Either sample from the `train` set, the `val` set, or the
                      `test` set.
        :return: A tuple of numpy array, the first item being the data and the
                 second the associated targets.
        """
        if split == "train":
            x, y, t = self.train_data
        else:
            x, y, t = self.test_data

        indexes = np.where(t == ind_task)[0]
        selected_x = x[indexes]
        selected_y = y[indexes]

        if self.cl_dataset.need_class_remapping:
            # A remapping of the class ids is done to handle some special cases
            # like PermutedMNIST or RotatedMNIST.
            selected_y = self.cl_dataset.class_remapping(selected_y)

        return selected_x, selected_y


    def _select_data_by_classes(self, min_class_id: int, max_class_id: int, split: str="train"):
        """Selects a subset of the whole data for a given set of classes.

        :param min_class_id: The minimum class id.
        :param max_class_id: The maximum class id.
        :param split: Either sample from the `train` set, the `val` set, or the
                      `test` set.
        :return: A tuple of numpy array, the first item being the data and the
                 second the associated targets.
        """
        if split == "train":
            x, y = self.train_data
        else:
            x, y = self.test_data

        indexes = np.where(np.logical_and(y >= min_class_id, y < max_class_id))[0]
        selected_x = x[indexes]
        selected_y = y[indexes]

        if self.cl_dataset.need_class_remapping:
            # A remapping of the class ids is done to handle some special cases
            # like PermutedMNIST or RotatedMNIST.
            selected_y = self.cl_dataset.class_remapping(selected_y)

        return selected_x, selected_y

