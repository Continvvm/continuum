from typing import Callable, List, Optional

import numpy as np
from torchvision import transforms

from continuum.datasets import _ContinuumDataset
from continuum.scenarios import InstanceIncremental
from continuum.tasks import TaskSet, TaskType


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
        if incremental_transformations is None:
            raise ValueError("For this scenario a list transformation should be set")

        if cl_dataset.data_type == TaskType.H5:
            raise NotImplementedError("TransformationIncremental are not compatible yet with h5 files.")

        self.inc_trsf = incremental_transformations
        #self._nb_tasks = self._setup(nb_tasks)
        self.shared_label_space = shared_label_space

        super().__init__(
            cl_dataset=cl_dataset, nb_tasks=nb_tasks, transformations=base_transformations
        )

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

        :param task_index: The unique index of a task, between 0 and len(loader) - 1. Or it could
                           be a list or a numpy array or even a slice.
        :return: A train PyTorch's Datasets.
        """
        x, y, _ = self.dataset

        if isinstance(task_index, slice):
            # Convert a slice to a list and respect the Python's advanced indexing conventions
            start = task_index.start if task_index.start is not None else 0
            stop = task_index.stop if task_index.stop is not None else len(self) + 1
            step = task_index.step if task_index.step is not None else 1
            task_index = list(range(start, stop, step))
            if len(task_index) == 0:
                raise ValueError(f"Invalid slicing resulting in no data (start={start}, end={stop}, step={step}).")
        elif isinstance(task_index, np.ndarray):
            task_index = list(task_index)
        elif isinstance(task_index, int):
            task_index = [task_index]
        else:
            raise TypeError(f"Invalid type of task index {type(task_index).__name__}.")

        task_index = set([_handle_negative_indexes(ti, len(self)) for ti in task_index])

        t = np.concatenate([
            (np.ones(len(x)) * ti).astype(np.int32) for ti in task_index
        ])
        x = np.concatenate([
            x for _ in range(len(task_index))
        ])
        if self.shared_label_space:
            y = np.concatenate([
                y for _ in range(len(task_index))
            ])
        else:
            # Different transformations have different labels even though
            # the original images were the same
            y = np.concatenate([
                y + ti * self.num_classes_per_task for ti in task_index
            ])

        trsf = [  # Non-used tasks have a None trsf
            self.get_task_transformation(ti)
            if ti in task_index else None
            for ti in range(len(self))
        ]

        return TaskSet(x, y, t, trsf, data_type=self.cl_dataset.data_type)


def _handle_negative_indexes(index: int, total_len: int) -> int:
    while index < 0:
        index += total_len
    return index
