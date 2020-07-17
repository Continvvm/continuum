import abc
from typing import Callable, List, Union

import numpy as np
from torchvision import transforms

from continuum.datasets import _ContinuumDataset
from continuum.task_set import TaskSet


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
            base_transformations: List[Callable] = None
    ) -> None:

        self.cl_dataset = cl_dataset
        self._nb_tasks = nb_tasks

        if base_transformations is None:
            base_transformations = self.cl_dataset.transformations
        self.trsf = transforms.Compose(base_transformations)

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

    def __next__(self) -> TaskSet:
        """An iteration/task in the for loop."""
        if self._counter >= len(self):
            raise StopIteration
        task = self[self._counter]
        self._counter += 1
        return task

    def __getitem__(self, task_index: Union[int, slice]):
        """Returns a task by its unique index.

        :param task_index: The unique index of a task. As for List, you can use
                           indexing between [0, len], negative indexing, or
                           even slices.
        :return: A train PyTorch's Datasets.
        """
        _x, _y, _t = self._select_data_by_task(task_index)
        return TaskSet(_x, _y, _t, self.trsf, data_type=self.cl_dataset.data_type)

    def _select_data_by_task(self, task_index: Union[int, slice]):
        """Selects a subset of the whole data for a given task.

        :param task_index: The unique index of a task. As for List, you can use
                           indexing between [0, len], negative indexing, or
                           even slices.
        :return: A tuple of numpy array, the first item being the data and the
                 second the associated targets.
        """
        x, y, t = self.dataset  # type: ignore

        if isinstance(task_index, slice):
            start = task_index.start or 0
            stop = task_index.stop or len(self) + 1
            step = task_index.step or 1
            task_indexes = list(range(start, stop, step))
            task_indexes = [
                t if t >= 0 else _handle_negative_indexes(t, len(self)) for t in task_indexes
            ]
            indexes = np.where(np.isin(t, task_indexes))[0]
        else:
            if task_index < 0:
                task_index = _handle_negative_indexes(task_index, len(self))
            indexes = np.where(t == task_index)[0]
        selected_x = x[indexes]
        selected_y = y[indexes]
        selected_t = t[indexes]

        if self.cl_dataset.need_class_remapping:  # TODO: to remove with TransformIncremental
            # A remapping of the class ids is done to handle some special cases
            # like PermutedMNIST or RotatedMNIST.
            selected_y = self.cl_dataset.class_remapping(selected_y)

        return selected_x, selected_y, selected_t


def _handle_negative_indexes(index: int, total_len: int) -> int:
    while index < 0:
        index += total_len
    return index
