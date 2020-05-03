import abc
from typing import Callable, List, Tuple

import numpy as np
from torchvision import transforms

from clloader.datasets import _ContinuumDataset
from clloader.task_set import TaskSet


class _BaseCLLoader(abc.ABC):
    """Abstract loader.

    DO NOT INSTANTIATE THIS CLASS.

    :param cl_dataset: A Continuum dataset.
    :param nb_tasks: The number of tasks to do.
    :param train_transformations: The PyTorch transformations exclusive to the
                                  train set.
    :param common_transformations: The PyTorch transformations common to the
                                   train set and the test set.
    :param train: Boolean flag whether to use the train or test subset.
    """

    def __init__(
        self,
        cl_dataset: _ContinuumDataset,
        nb_tasks: int,
        train_transformations: List[Callable] = None,
        common_transformations: List[Callable] = None,
        train=True
    ) -> None:

        self.cl_dataset = cl_dataset
        self._nb_tasks = nb_tasks

        if train_transformations is None:
            train_transformations = []
        if common_transformations is None:
            common_transformations = self.cl_dataset.transformations

        if len(common_transformations) == 0:
            self.train_trsf, self.test_trsf = None, None
        else:
            self.train_trsf = transforms.Compose(train_transformations + common_transformations)
            self.test_trsf = transforms.Compose(common_transformations)
        self.train = train

    @abc.abstractmethod
    def _setup(self, nb_tasks: int) -> int:
        raise NotImplementedError

    @property
    def nb_classes(self) -> int:
        """Total number of classes in the whole continual setting."""
        return len(np.unique(self.dataset[1]))  # type: ignore

    @property
    def nb_tasks(self) -> int:
        """Number of tasks in the whole continual setting."""
        return len(self)

    def __len__(self) -> int:
        """Returns the number of tasks.

        :return: Number of tasks.
        """
        return self._nb_tasks

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
        :return: A train PyTorch's Datasets.
        """
        data = self._select_data_by_task(task_index)
        return TaskSet(
            *data,
            self.train_trsf if self.train else self.test_trsf,
            data_type=self.cl_dataset.data_type
        )

    def _select_data_by_task(self, ind_task: int):
        """Selects a subset of the whole data for a given task.

        :param ind_task: task index
        :return: A tuple of numpy array, the first item being the data and the
                 second the associated targets.
        """
        x_, y_, t_ = self.dataset  # type: ignore

        indexes = np.where(t_ == ind_task)[0]
        selected_x = x_[indexes]
        selected_y = y_[indexes]

        if self.cl_dataset.need_class_remapping:  # TODO: to remove with TransformIncremental
            # A remapping of the class ids is done to handle some special cases
            # like PermutedMNIST or RotatedMNIST.
            selected_y = self.cl_dataset.class_remapping(selected_y)

        return selected_x, selected_y
