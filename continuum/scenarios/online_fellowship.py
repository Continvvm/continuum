from typing import Callable, List, Union
from torchvision import transforms
import numpy as np

from continuum.datasets import _ContinuumDataset, InMemoryDataset
from continuum.scenarios import _BaseScenario
from continuum.tasks import TaskSet


class OnlineFellowship(_BaseScenario):
    """A scenario to create large fellowship and load them one by one. No fancy stream, one cl_dataset = one task.
    The advantage toward using a Fellowship dataset is that the datasets might have different data_type.

    :param cl_datasets: A list of continual dataset.
    :param transformations: A list of transformations applied to all tasks. If
                            it's a list of list, then the transformation will be
                            different per task.
    :param update_labels: if true we update labels values such as not having same classes in different tasks.
    :param list_dict_args: Parameters to use to instanciate datasets.
    """

    def __init__(
            self,
            cl_datasets: List[_ContinuumDataset],
            transformations: Union[List[Callable], List[List[Callable]]] = None,
            update_labels=True,
            list_dict_args=[{"data_path": ".", "train": True, "download": False}]
    ) -> None:
        self.cl_datasets = cl_datasets
        self.update_labels = update_labels
        self.list_dict_args = list_dict_args
        self.transformations = transformations
        assert len(self.list_dict_args) == 1 or len(self.list_dict_args) == len(cl_datasets)
        # init with first task
        if isinstance(cl_datasets[0], InMemoryDataset):
            self.cl_dataset = cl_datasets[0]
        else:
            self.cl_dataset = cl_datasets[0](**self._get_args(0))

        if isinstance(self.trsf, list):
            # if we have a list of transformations, it should be a transformation per cl_dataset
            assert len(transformations) == self._nb_tasks

        trsf_0 = self.get_trsf(task_index=0)

        self.label_trsf = []
        self.list_classes_sofar = []
        self.list_task_sofar = []

        super().__init__(cl_dataset=self.cl_dataset, nb_tasks=1, transformations=trsf_0)
        self._nb_tasks = len(cl_datasets)

    def _get_args(self, ind_task):
        if len(self.list_dict_args) == 1:
            return self.list_dict_args[0]
        else:
            return self.list_dict_args[ind_task]

    def _get_trsf(self, ind_task):
        if isinstance(self.transformations, list):
            trsf = self.transformations[ind_task]
        else:
            trsf = self.transformations
        return trsf

    def _get_label_trsf(self, ind_task):
        label_trsf = None
        if self.update_labels:
            if not (ind_task in self.list_task_sofar):
                raise AssertionError("The future can not be indexed, task should be visited in the right order.")
            label_trsf = self.label_trsf[ind_task]
        return label_trsf

    @property
    def nb_tasks(self) -> int:
        """Number of tasks in the whole continual setting."""
        return self._nb_tasks

    @property
    def nb_classes(self) -> int:
        """Number of classes in the whole continual setting."""
        return len(self.classes)

    @property
    def classes(self) -> List:
        """List of classes in the whole continual setting."""
        print("classes seen so far. You can not know the total number of class before visiting all tasks.")
        return np.unique(self.list_classes_sofar)

    def __getitem__(self, task_index: Union[int, slice]):
        """Returns a task by its unique index.

        :param task_index: The unique index of a task. As for List, you can use
                           indexing between [0, len], negative indexing, or
                           even slices.
        :return: A train PyTorch's Datasets.
        """
        if isinstance(task_index, slice):
            raise NotImplementedError(
                f"You cannot select multiple task ({task_index}) on OnlineFellowship scenario yet"
            )

        if self.update_labels:
            # we just shift the label number by the nb of classes seen so far
            self.label_trsf.append(transforms.Lambda(lambda x: x + len(self.list_classes_sofar)))

        label_trsf = self._get_label_trsf(task_index)

        if isinstance(self.cl_datasets[task_index], InMemoryDataset):
            self.cl_dataset = self.cl_datasets[task_index]
        else:
            self.cl_dataset = self.cl_datasets[task_index](**self._get_args(task_index))

        x, y, _ = self.cl_dataset.get_data()
        self.list_classes_sofar += np.unique(y)
        self.list_task_sofar.append(task_index)

        t = np.ones(len(y)) * task_index

        return TaskSet(
            x, y, t,
            trsf=self._get_trsf(task_index),
            target_trsf=label_trsf,
            data_type=self.cl_dataset.data_type,
            bounding_boxes=self.cl_dataset.bounding_boxes
        )


def _setup(self, nb_task):
    return nb_task
