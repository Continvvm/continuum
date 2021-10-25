from typing import Callable, List, Union
from torchvision import transforms
import numpy as np

from continuum.datasets import _ContinuumDataset, InMemoryDataset
from continuum.scenarios import _BaseScenario
from continuum.tasks import TaskSet, TaskType
from continuum.transforms.segmentation import Compose as SegmentationCompose


class OnlineFellowship(_BaseScenario):
    """A scenario to create large fellowship and load them one by one. No fancy stream, one cl_dataset = one task.
    The advantage toward using a Fellowship dataset is that the datasets might have different data_type. It is recommanded
    to not use inMemoryDataset in OnlineFellowship to not make the scenario to heavy in memory.

    :param cl_datasets: A list of continual dataset already instanciate.
    :param transformations: A list of transformations applied to all tasks. If
                            it's a list of list, then the transformation will be
                            different per task.
    :param update_labels: if true we update labels values such as not having same classes in different tasks.
    """

    def __init__(
            self,
            cl_datasets: List[_ContinuumDataset],
            transformations: Union[List[Callable], List[List[Callable]]] = None,
            update_labels=True
    ) -> None:
        self.cl_datasets = cl_datasets
        self.update_labels = update_labels

        # init with first task
        self.cl_dataset = cl_datasets[0]
        trsf_0 = self._get_trsf(ind_task=0, transformations=transformations, compose=False)
        super().__init__(cl_dataset=self.cl_dataset, nb_tasks=1, transformations=trsf_0)

        self.trsf = transformations
        self.transformations = transformations
        self._nb_tasks = len(cl_datasets)
        self._setup(nb_tasks=self._nb_tasks)

    def _setup(self, nb_tasks: int) -> int:

        # small check on self.trsf
        if self.trsf is not None and isinstance(self.trsf[0], list):
            # We have list of list of callable, where each sublist is dedicated to
            # a task
            assert len(self.trsf) == len(self.cl_datasets), \
                print("The transformations is not set correctly. It should be: "
                      "A list of transformations applied to all tasks. "
                      "Or a list of list of size nb_task, with one transformation list per task.")

        # we count classes and create label transform function if necessary (i.e. for update_labels=True).
        self.label_trsf = []
        self.list_unique_classes = np.zeros(0)
        for dataset in self.cl_datasets:
            x, y, _ = dataset.get_data()
            classes = np.unique(y)
            assert np.all(classes == np.arange(len(classes))), print("Classes are not annotated correctly, they are"
                                                                     "expected to be annotated continuously from 0 to N-1.")

            if self.update_labels:
                # we just shift the label number by the nb of classes seen so far
                self.label_trsf.append(transforms.Lambda(lambda x: x + len(self.list_unique_classes)))
                # shift classes indexes for self.list_unique_classes
                classes = classes + len(self.list_unique_classes)

            self.list_unique_classes = np.unique(np.concatenate([self.list_unique_classes, classes]))

    def _get_trsf(self, ind_task, transformations, compose=True):
        """" Manage data transformation."""

        if transformations is None:
            # then we set the default dataset transformations if any
            transformations = self.cl_datasets[ind_task].transformations
        if transformations is not None and isinstance(transformations[0], list):
            # we take the transformations specific to the task ind_task
            transformations = transformations[ind_task]

        if compose:
            # convert the list into a composer
            if self.cl_datasets[ind_task].data_type == TaskType.SEGMENTATION:
                composer = SegmentationCompose
            else:
                composer = transforms.Compose
            transformations = composer(transformations)

        return transformations

    def _get_label_trsf(self, ind_task):
        """" Manage data label transformation. Necessary especially if update_labels is True. """
        label_trsf = None
        if self.update_labels:
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
        return np.unique(self.list_unique_classes)

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

        self.cl_dataset = self.cl_datasets[task_index]
        x, y, _ = self.cl_dataset.get_data()
        t = np.ones(len(y)) * task_index

        return TaskSet(
            x, y, t,
            trsf=self._get_trsf(task_index, self.transformations),
            target_trsf=self._get_label_trsf(task_index),
            data_type=self.cl_dataset.data_type,
            bounding_boxes=self.cl_dataset.bounding_boxes
        )
