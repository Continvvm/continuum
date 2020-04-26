from typing import Callable, List, Union
from copy import copy

import numpy as np

from clloader.datasets import BaseDataset

from clloader import BaseCLLoader


class ClassIncremental(BaseCLLoader):
    """Continual Loader, generating datasets for the consecutive tasks.

    :param cl_dataset: A continual dataset.
    :param nb_tasks: The scenario number of tasks
    :param increment: Either number of classes per task, or a list specifying for
                      every task the amount of new classes.
    :param initial_increment: A different task size applied only for the first task.
                              Desactivated if `increment` is a list.
    :param train_transformations: A list of data augmentation applied to the train set.
    :param common_transformations: A list of transformations applied to both the
                                   the train set and the test set. i.e. normalization,
                                   resizing, etc.
    :param class_order: An optional custom class order, used for NC.
    """
    """Continual Loader, generating datasets for the consecutive tasks.

    :param cl_dataset: A continual dataset.
    :param increment: Either number of classes per task, or a list specifying for
                      every task the amount of new classes.
    :param initial_increment: A different task size applied only for the first task.
                              Desactivated if `increment` is a list.
    :param train_transformations: A list of data augmentation applied to the train set.
    :param common_transformations: A list of transformations applied to both the
                                   the train set and the test set. i.e. normalization,
                                   resizing, etc.
    :param class_order: An optional custom class order, used for NC.
    """

    def __init__(
            self,
            cl_dataset: BaseDataset,
            nb_tasks: int =0,
            increment: Union[List[int], int]=0,
            initial_increment: int = 0,
            train_transformations: List[Callable] = None,
            common_transformations: List[Callable] = None,
            train=True,
            class_order=None
    ) -> None:

        super().__init__(cl_dataset=cl_dataset,
                         nb_tasks=nb_tasks,
                         train_transformations=train_transformations,
                         common_transformations=common_transformations,
                         train=train)

        self._setup_class_incremental(increment, initial_increment, class_order)

    def _setup_class_incremental(self, increment: Union[List[int], int],
                                 initial_increment: int,
                                 class_order: Union[None, List[int]] = None) -> None:

        unique_classes = np.unique(self.dataset[1])  # search for unique classes

        self.class_order = class_order or self.cl_dataset.class_order or list(
            range(len(unique_classes))
        )

        if len(np.unique(self.class_order)) != len(self.class_order):
            raise ValueError(f"Invalid class order, duplicates found: {self.class_order}.")

        mapper = np.vectorize(lambda x: self.class_order.index(x))
        new_y = mapper(self.dataset[1])

        # Increments setup
        self.class_order = np.array(self.class_order)
        self.increments = self._define_increments(increment, initial_increment)

        # compute task label
        t_ = self._set_task_labels(new_y)

        # Dataset with task label
        self.dataset = (self.dataset[0], new_y, t_)  # (data, class label, task label)

    def _set_task_labels(self, y: np.ndarray) -> np.ndarray:
        """
        For each data point, defines a task associated with the data
        :param y: label tensor
        :param increments: increments contains information about classes per tasks
        :return: tensor of task label
        """

        t = copy(y)  # task label as same size as y
        for task_index, increment in enumerate(self.increments):
            max_class = sum(self.increments[:task_index + 1])
            min_class = sum(self.increments[:task_index])  # 0 when task_index == 0.

            indexes = np.where(np.logical_and(y >= min_class, y < max_class))[0]
            t[indexes] = task_index
        return t

    def _define_increments(self, increment: Union[List[int], int],
                           initial_increment: int) -> List[int]:
        if isinstance(increment, list):

            # Check if the total number of classes is compatible between increment list and self.nb_classes
            if not sum(increment) == self.nb_classes:
                raise Exception(
                    "The increment list is not compatible with the number of classes"
                )

            increments = increment
        else:
            increments = []
            if initial_increment:
                increments.append(initial_increment)

            nb_tasks = (self.nb_classes - initial_increment) / increment
            if not nb_tasks.is_integer():
                raise Exception(
                    "The tasks won't have an equal number of classes"
                    f" with {len(self.class_order)} and increment {increment}"
                )
            increments.extend([increment for _ in range(int(nb_tasks))])

        return increments

    def get_original_targets(self, targets: np.ndarray) -> np.ndarray:
        """Returns the original targets not changed by the custom class order.

        :param targets: An array of targets, as provided by the task datasets.
        :return: An array of targets, with their original values.
        """
        return self.class_order[targets]

    def _select_data_by_classes(self, min_class_id: int, max_class_id: int):
        """Selects a subset of the whole data for a given set of classes.

        :param min_class_id: The minimum class id.
        :param max_class_id: The maximum class id.
        :return: A tuple of numpy array, the first item being the data and the
                 second the associated targets.
        """
        x_, y_, t_ = self.dataset

        indexes = np.where(np.logical_and(y_ >= min_class_id, y_ < max_class_id))[0]
        selected_x = x_[indexes]
        selected_y = y_[indexes]

        if self.cl_dataset.need_class_remapping:
            # A remapping of the class ids is done to handle some special cases
            # like PermutedMNIST or RotatedMNIST.
            selected_y = self.cl_dataset.class_remapping(selected_y)

        return selected_x, selected_y
