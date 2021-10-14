from typing import Callable, List, Union
from torchvision import transforms
import numpy as np

from continuum.datasets import _ContinuumDataset, InMemoryDataset
from continuum.scenarios import _BaseScenario
from continuum.tasks import TaskSet


class OnlineFellowship(_BaseScenario):
    """A scenario to create large fellowship and load them one by one. No fancy stream, one cl_dataset = one task

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
            list_dict_args=[{"data_path":".", "train": True, "download": False}]
    ) -> None:
        self.cl_datasets = cl_datasets
        self.update_labels = update_labels
        self.list_dict_args = list_dict_args
        assert len(self.list_dict_args)==1 or len(self.list_dict_args)==len(cl_datasets)
        # init with first task
        if isinstance(cl_datasets[0], InMemoryDataset):
            self.cl_dataset = cl_datasets[0]
        else:
            self.cl_dataset = cl_datasets[0](**self._get_args(0))
        super().__init__(cl_dataset=self.cl_dataset, nb_tasks=1, transformations=transformations)
        self._nb_tasks = len(cl_datasets)

        if isinstance(self.trsf, list):
            # if we have a list of transformations, it should be a transformation per cl_dataset
            assert len(transformations) == self._nb_tasks

        # We count total number of classes here and define label_trsf if necessary if (update_labels==True)
        _tot_num_classes = 0
        label_trsf = []
        self.list_classes = []
        for task_ind, cl_dataset in enumerate(cl_datasets):
            if isinstance(cl_dataset, InMemoryDataset):
                dataset = cl_dataset
            else:
                dataset = cl_dataset(**self._get_args(task_ind))
            if self.update_labels:
                # we just shift the label number by the nb of classes seen so far
                label_trsf.append(transforms.Lambda(lambda x: x + _tot_num_classes))
                self.list_classes += (list(np.array(dataset.classes)+_tot_num_classes))
            else:
                self.list_classes += dataset.classes
            _tot_num_classes += dataset.nb_classes

        if not self.update_labels:
            label_trsf = None
        self.target_trsf = label_trsf

    def _get_args(self, ind_task):
        if len(self.list_dict_args) == 1:
            return self.list_dict_args[0]
        else:
            return self.list_dict_args[ind_task]


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
        if isinstance(self.cl_datasets[task_index], InMemoryDataset):
            self.cl_dataset = self.cl_datasets[task_index]
        else:
            self.cl_dataset = self.cl_datasets[task_index](**self._get_args(task_index))
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