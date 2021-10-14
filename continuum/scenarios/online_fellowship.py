from typing import Callable, List, Union
from torchvision import transforms
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
            transformations: Union[List[Callable], List[List[Callable]]] = None,
            update_labels=True
    ) -> None:
        self._nb_tasks = len(cl_datasets)
        self.cl_datasets = cl_datasets
        self.update_labels = update_labels
        # init with first task
        super().__init__(cl_dataset=cl_datasets[0], nb_tasks=1, transformations=transformations)

        if isinstance(self.trsf, list):
            # if we have a list of transformations, it should be a transformation per cl_dataset
            assert len(transformations) == self._nb_tasks

        _tot_num_classes = 0
        label_trsf = []
        self.list_classes = []
        for task_ind, cl_dataset in enumerate(cl_datasets):
            if self.update_labels:
                # we just shift the label number by the nb of classes seen so far
                label_trsf.append(transforms.Lambda(lambda x: x + _tot_num_classes))
                self.list_classes += (list(np.array(cl_dataset.classes)+_tot_num_classes))
            else:
                self.list_classes += cl_dataset.classes
            _tot_num_classes += cl_dataset.nb_classes

        if not self.update_labels:
            label_trsf = None
        self.target_trsf = label_trsf


    @property
    def nb_tasks(self) -> int:
        """Number of tasks in the whole continual setting."""
        return self._nb_tasks

    @property
    def nb_classes(self) -> int:
        """Number of tasks in the whole continual setting."""
        return len(self.classes)

    @property
    def classes(self) -> List:
        """Number of tasks in the whole continual setting."""
        return np.unique(self.list_classes)

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
            target_trsf=self.target_trsf[task_index] if isinstance(self.target_trsf, list) else self.target_trsf,
            data_type=self.cl_dataset.data_type,
            bounding_boxes=self.cl_dataset.bounding_boxes
        )

    def _setup(self, nb_task):
        return nb_task