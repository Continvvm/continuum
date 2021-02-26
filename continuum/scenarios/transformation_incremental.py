from typing import Callable, List

import numpy as np
from torchvision import transforms

from continuum.datasets import _ContinuumDataset
from continuum.scenarios import InstanceIncremental
from continuum.tasks import TaskSet


class TransformationIncremental(InstanceIncremental):
    """Continual Loader, generating datasets for the consecutive tasks.

    Scenario: Every task contains the same data with different transformations.
    It is a cheap way to create instance incremental scenarios.
    Moreover, it is easier to analyse what algorithms forget or not.
    Classic transformation incremental scenarios are "permutations" and "rotations".

    :param cl_dataset: A continual dataset.
    :param incremental_transformations: list of transformations to apply to specific tasks
    :param base_transformations: List of transformation to apply to all tasks.
    :param shared_label_space: If true same data with different transformation have same label
    """

    def __init__(
            self,
            cl_dataset: _ContinuumDataset,
            incremental_transformations: List[List[Callable]],
            base_transformations: List[Callable] = None,
            shared_label_space=True
    ):
        nb_tasks = len(incremental_transformations)
        super().__init__(
            cl_dataset=cl_dataset, nb_tasks=nb_tasks, transformations=base_transformations
        )
        if incremental_transformations is None:
            raise ValueError("For this scenario a list transformation should be set")

        self.inc_trsf = incremental_transformations
        self._nb_tasks = self._setup(nb_tasks)
        self.shared_label_space = shared_label_space
        self.num_classes_per_task = len(np.unique(self.dataset[1]))  # the num of classes is the same for all task is this scenario

    @property
    def nb_classes(self) -> int:
        """Total number of classes in the whole continual setting."""
        if self.shared_label_space:
            nb_classes = len(np.unique(self.dataset[1]))
        else:
            nb_classes = len(np.unique(self.dataset[1])) * self._nb_tasks
        return nb_classes

    def get_task_transformation(self, task_index):
        return transforms.Compose(self.inc_trsf[task_index] + self.trsf.transforms)

    def update_task_indexes(self, task_index):
        new_t = np.ones(len(self.dataset[1])) * task_index
        self.dataset = (self.dataset[0], self.dataset[1], new_t)

    def update_labels(self, task_index):
        # wrong
        # new_y = self.dataset[1] + task_index * self.num_classes_per_task
        # we update incrementally then update is simply:
        if task_index > 0:
            new_y = self.dataset[1] + self.num_classes_per_task
            self.dataset = (self.dataset[0], new_y, self.dataset[2])

    def __getitem__(self, task_index):
        """Returns a task by its unique index.

        :param task_index: The unique index of a task, between 0 and len(loader) - 1.
        :return: A train PyTorch's Datasets.
        """
        if isinstance(task_index, slice):
            raise ValueError(
                "Incremental training based on transformations "
                "does not support slice, please provide only integer."
            )
        elif task_index < 0:  # Support for negative index, e.g. -1 == last
            while task_index < 0:
                task_index += len(self)

        self.update_task_indexes(task_index)
        if not self.shared_label_space:
            self.update_labels(task_index)
        x, y, t, _ = self._select_data_by_task(task_index)
        trsf = self.get_task_transformation(task_index)

        return TaskSet(x, y, t, trsf, data_type=self.cl_dataset.data_type)
