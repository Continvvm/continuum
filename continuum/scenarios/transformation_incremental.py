from typing import Callable, List

import numpy as np

from clloader.task_set import TaskSet
from clloader.datasets import _ContinuumDataset
from clloader.scenarios import InstanceIncremental

from torchvision import transforms

class TransformationIncremental(InstanceIncremental):
    """Continual Loader, generating datasets for the consecutive tasks.
    Scenario: Mode incremental scenario is a new instance scenario where we explore the distribution mode by mode.
              For example rotation mnist, is a exploration of the distribution by rotation angles, each angle can be
              seen as a mode of the distribution. Same for permutMnist, mode=permutations space.

    :param cl_dataset: A continual dataset.
    :param increment: Either number of classes per task, or a list specifying for
                      every task the amount of new classes.
    :param initial_increment: A different task size applied only for the first task.
                              Desactivated if `increment` is a list.
    :param train_transformations: A list of data augmentation applied to the train set.
    :param common_transformations: A list of transformations applied to both the
                                   the train set and the test set. i.e. normalization,
                                   resizing, etc.
    :param incremental_transformations: A list of transformations specific to each tasks. e.g. rotations, permutations
    """

    def __init__(
        self,
        cl_dataset: _ContinuumDataset,
        nb_tasks: int,
        incremental_transformations: List[List[Callable]],
        base_transformations: List[Callable] = None
    ):
        super().__init__(
            cl_dataset=cl_dataset,
            nb_tasks=nb_tasks,
            base_transformations=base_transformations
        )
        if incremental_transformations is None:
            raise ValueError("For this scenario a list transformation should be set")

        if nb_tasks != len(incremental_transformations):
            raise ValueError("The number of tasks is not equal to the number of transformation")

        self.inc_trsf = incremental_transformations
        self._nb_tasks = self._setup(nb_tasks)

    def get_task_index(self, nb_tasks, y):
        # all tasks have all labels, only the transformation change
        return y

    def get_task_transformation(self, task_index):
        return transforms.Compose(self.inc_trsf[task_index]+self.trsf.transforms)


    def update_task_indexes(self, task_index):
        new_t_ = np.ones(len(self.dataset[1]))*task_index
        self.dataset = (self.dataset[0], self.dataset[1], new_t_)

    def __getitem__(self, task_index):
        """Returns a task by its unique index.

        :param task_index: The unique index of a task, between 0 and len(loader) - 1.
        :return: A train PyTorch's Datasets.
        """
        self.update_task_indexes(task_index)
        train = self._select_data_by_task(task_index)
        trsf = self.get_task_transformation(task_index)
        train_dataset = TaskSet(*train, trsf, data_type=self.cl_dataset.data_type)

        return train_dataset