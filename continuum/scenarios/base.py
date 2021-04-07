import abc
from typing import Callable, List, Union

import numpy as np
from torchvision import transforms

from continuum.datasets import _ContinuumDataset
from continuum.tasks import TaskSet
from continuum.transforms.segmentation import Compose as SegmentationCompose


class _BaseScenario(abc.ABC):
    """Abstract loader.

    DO NOT INSTANTIATE THIS CLASS.

    :param cl_dataset: A Continuum dataset.
    :param nb_tasks: The number of tasks to do.
    :param transformations: The PyTorch transformations.
    :param train: Boolean flag whether to use the train or test subset.
    """

    def __init__(
            self,
            cl_dataset: _ContinuumDataset,
            nb_tasks: int,
            transformations: List[Callable] = None
    ) -> None:

        self.cl_dataset = cl_dataset
        self._nb_tasks = nb_tasks

        if transformations is None:
            transformations = self.cl_dataset.transformations
        if self.cl_dataset.data_type == "segmentation":
            self.trsf = SegmentationCompose(transformations)
        else:
            self.trsf = transforms.Compose(transformations)

    @abc.abstractmethod
    def _setup(self, nb_tasks: int) -> int:
        raise NotImplementedError

    @property
    def train(self) -> bool:
        """Returns whether we are in training or testing mode.

        This property is dependent on the dataset, not the actual scenario.
        """
        return self.cl_dataset.train

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
        x, y, t, _ = self._select_data_by_task(task_index)
        return TaskSet(
            x, y, t,
            trsf=self.trsf,
            data_type=self.cl_dataset.data_type,
            bounding_boxes=self.cl_dataset.bounding_boxes
        )

    def _select_data_by_task(
        self,
        task_index: Union[int, slice]
    ) -> Union[np.ndarray, np.ndarray, np.ndarray, Union[int, List[int]]]:
        """Selects a subset of the whole data for a given task.

        This class returns the "task_index" in addition of the x, y, t data.
        This task index is either an integer or a list of integer when the user
        used a slice. We need this variable when in segmentation to disangle
        samples with multiple task ids.

        :param task_index: The unique index of a task. As for List, you can use
                           indexing between [0, len], negative indexing, or
                           even slices.
        :return: A tuple of numpy array being resp. (1) the data, (2) the targets,
                 (3) task ids, and (4) the actual task required by the user.
        """
        x, y, t = self.dataset  # type: ignore

        if isinstance(task_index, slice):
            start = task_index.start if task_index.start is not None else 0
            stop = task_index.stop if task_index.stop is not None else len(self) + 1
            step = task_index.step if task_index.step is not None else 1
            task_index = list(range(start, stop, step))
            if len(task_index) == 0:
                raise ValueError(f"Invalid slicing resulting in no data (start={start}, end={stop}, step={step}).")
            task_index = [
                t if t >= 0 else _handle_negative_indexes(t, len(self)) for t in task_index
            ]
            if len(t.shape) == 2:
                indexes = np.unique(np.where(t[:, task_index] == 1)[0])
            else:
                indexes = np.where(np.isin(t, task_index))[0]
        else:
            if task_index < 0:
                task_index = _handle_negative_indexes(task_index, len(self))

            if len(t.shape) == 2:
                indexes = np.where(t[:, task_index] == 1)[0]
            else:
                indexes = np.where(t == task_index)[0]

        selected_x = x[indexes]
        selected_y = y[indexes]
        selected_t = t[indexes]

        if self.cl_dataset.need_class_remapping:  # TODO: to remove with TransformIncremental
            # A remapping of the class ids is done to handle some special cases
            # like PermutedMNIST or RotatedMNIST.
            selected_y = self.cl_dataset.class_remapping(selected_y)

        return selected_x, selected_y, selected_t, task_index


def _handle_negative_indexes(index: int, total_len: int) -> int:
    while index < 0:
        index += total_len
    return index
