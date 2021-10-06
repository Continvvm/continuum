from typing import Callable, List, Union

import numpy as np

from continuum.datasets import _ContinuumDataset
from continuum.scenarios import _BaseScenario
from continuum.tasks import TaskSet


class OnlineFellowship(_BaseScenario):
    """A scenario to create large fellowship and load them one by one. No fancy stream, one cl_dataset = one task

    :param cl_datasets: A list of continual dataset.
    :param transformations: A list of transformations applied to all tasks. If
                            it's a list of list, then the transformation will be
                            different per task.
    """

    def __init__(
            self,
            cl_datasets: List[_ContinuumDataset],
            nb_tasks: int = 0,
            transformations: Union[List[Callable], List[List[Callable]]] = None,
            class_order: Union[List[int], None] = None
    ) -> None:
        self._nb_tasks = len(cl_datasets)
        self.cl_datasets = cl_datasets
        # init with first task
        super().__init__(cl_dataset=cl_datasets[0], nb_tasks=1, transformations=transformations)

        if isinstance(self.trsf, list):
            # if we have a list of transformations, it should be a transformation per cl_dataset
            assert len(transformations) == self._nb_tasks

    @property
    def nb_tasks(self) -> int:
        """Number of tasks in the whole continual setting."""
        return self._nb_tasks

    def __getitem__(self, task_index: Union[int, slice]):
        """Returns a task by its unique index.

        :param task_index: The unique index of a task. As for List, you can use
                           indexing between [0, len], negative indexing, or
                           even slices.
        :return: A train PyTorch's Datasets.
        """
        if isinstance(task_index, slice):
            raise NotImplementedError(
                f"You cannot select multiple task ({task_index}) on OnlineFellowship yet"
            )
        self.cl_dataset = self.cl_datasets[task_index]
        x, y, _ = self.cl_dataset.get_data()
        t = np.ones(len(y)) * task_index

        return TaskSet(
            x, y, t,
            trsf=self.trsf[task_index] if isinstance(self.trsf, list) else self.trsf,
            data_type=self.cl_dataset.data_type,
            bounding_boxes=self.cl_dataset.bounding_boxes
        )
