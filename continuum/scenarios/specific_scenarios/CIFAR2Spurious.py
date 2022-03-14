from typing import Callable, List, Union

import numpy as np
import torch

from continuum.datasets import CIFAR10

from continuum.scenarios import InstanceIncremental
from copy import copy


class CIFAR2Spurious(InstanceIncremental):
    """CIFAR2Spurious scenario create a sequence of tasks with spurious features: square of colors.
    We use CIFAR10 dataset and reassign class label to transportation means vs not a trasnportation means.
    [0-airplane, 1-automobile, 2-bird, 3-cat, 4-deer, 5-dog, 6-frog, 7-horse, 8-ship, 9-truck]
     become [0,0,1,1,1,1,1,0,0,0]

    Scenario: CIFAR2Spurious scenarios, use same classes for all task but with spurious features.
    Each task get a specific purious features, such as all tasks are different.
    :param cl_dataset: the instantiate CIFAR10 continual dataset.
    :param nb_tasks: The scenario's number of tasks. (
    /!\ if it is a test scenario we add automatically a supplementary task without spurious feature)
    :param seed: initialization seed for the random number generator.
    :param correlation: correlation between label and spurious feature, 1.0 full correlation, 0.0 no correlation.
     To lower correlation we just do not add spurious feature to a certain amount of data point.
    :param support: the amount of the original data that is inside each task, if it is 1.0 all data is in all tasks,
     0.4 we only keep the data of 40% of the original classes for one task (we ensure balance between class 0 and 1)
    """

    def __init__(
            self,
            cl_dataset: CIFAR10,
            nb_tasks: int = None,
            seed: int = 0,
            correlation: float = 1.0,
            support: float = 1.0,
            train=True
    ):
        assert nb_tasks is not None

        self.training = train
        self.correlation = correlation
        self.support = support

        assert support > 0.0 and support <= 1.0
        self.seed = seed
        super().__init__(
            cl_dataset=cl_dataset,
            nb_tasks=nb_tasks,
            random_seed = seed
        )

        assert type(cl_dataset).__name__ == "CIFAR10", print(type(cl_dataset).__name__)
        # means of transportation
        # old_labels = [airplane, automobile, , bird, cat, deer, dog, frog, horse, ship, truck]
        self.new_labels=torch.LongTensor([0,0,1,1,1,1,1,0,0,0])

        self.initial_nb_classes = len(np.unique(self.dataset[1]))
        self.image_size = 32

    @property
    def nb_tasks(self) -> int:
        """Number of tasks in the whole continual setting."""

        return self._nb_tasks

    @property
    def nb_classes(self):
        return len(self.classes)

    @property
    def classes(self):
        return np.unique(self.class_remapping(self.dataset[1]))

    def _generate_two_colors(self, ind_task):
        rng = np.random.RandomState(seed=self.seed + ind_task)
        colors = rng.choice(range(16), size=(2, 3)) * 16  # 16 * 16 = 256, we don't want to have too close colors

        return colors

    def class_remapping(self, y):
        return self.new_labels[y]

    def _data_transformation(self, x, labels, ind_task):
        """"Transform data for scenario purposes
         it can not be done with classical transform since the transformation depends on the label
         We convert [1,28,28] images into [3,28,28]
        """
        nb_samples = len(labels)
        data_task = copy(x)

        if (self.training or (ind_task < self.nb_tasks - 1)):

            centers = torch.randint(2, self.image_size - 2, (nb_samples, 2))

            colors = self._generate_two_colors(ind_task)
            for i in range(nb_samples):
                # uniform sampling between 0 and 1 to create correlation
                if np.random.uniform() < self.correlation:
                    square_size = 2  # random.choice([1, 2, 3])
                    center = centers[i, :]  # [5, 5]
                    label = labels[i]
                    color = colors[label]
                    x_min = max(0, center[0] - square_size)
                    x_max = min(self.image_size - 1, center[0] + square_size)
                    y_min = max(0, center[1] - square_size)
                    y_max = min(self.image_size - 1, center[1] + square_size)

                    data_task[i, x_min:x_max, y_min:y_max, 0] = color[0]
                    data_task[i, x_min:x_max, y_min:y_max, 1] = color[1]
                    data_task[i, x_min:x_max, y_min:y_max, 2] = color[2]

        return data_task

    def _setup(self, nb_tasks: int) -> int:
        x, y, t = self.cl_dataset.get_data()
        self.dataset = [x, y, t]

        if not self.training:
            # for test we add one more task with data without spurious feature
            nb_tasks += 1

        return nb_tasks

    def select_support(self, x, y, t, ind_task):
        """"
        Select a subset of the support. The selection is done based on original labels.
        If we decide to use a support of 0.5 we select data from only half of the original classes.
        """
        np.random.seed(self.seed + ind_task)
        rand_class_order = np.random.permutation(self.initial_nb_classes)

        nb_classes = int(np.ceil(self.initial_nb_classes * self.support / 2))

        # select nb_classes for both labels
        selected_classes_0 = rand_class_order[np.where(self.class_remapping(rand_class_order) == 0)[0][:nb_classes]]
        selected_classes_1 = rand_class_order[np.where(self.class_remapping(rand_class_order) == 1)[0][:nb_classes]]

        # concatenate selected classes
        selected_classes = np.concatenate([selected_classes_0, selected_classes_1])

        # get index in y vector to keep
        y_indexes = np.concatenate([np.where(y==value)[0] for value in selected_classes])
        # reoder everything
        y_indexes.sort()
        x = x[y_indexes]
        y = y[y_indexes]
        if t is not None:
            t = t[y_indexes]
        return x, y, t

    def _select_data_by_task(
            self,
            task_index: int
    ) -> Union[np.ndarray, np.ndarray, np.ndarray, Union[int, List[int]]]:
        """Selects a subset of the whole data for a given task.

        This class returns the "task_index" in addition of the x, y, t data.
        This task index is either an integer or a list of integer when the user
        used a slice. We need this variable when in segmentation to disentangle
        samples with multiple task ids.

        :param task_index: The unique index of a task. As for List, you can use
                           indexing between [0, len], negative indexing, or
                           even slices.
        :return: A tuple of numpy array being resp. (1) the data, (2) the targets,
                 (3) task ids, and (4) the actual task required by the user.
        """

        assert int(task_index) == task_index, print(f"this scenario does not support slicing: task_index {task_index} - {type(task_index)}")
        assert task_index < self.nb_tasks


        x, y, t = self.dataset

        # no selection if support is one
        if (self.support != 1.0):
            # no support selection if we are at the last test task
            if not ((not self.train) and (task_index == self.nb_tasks-1)):
                x, y, t = self.select_support(x, y, t, task_index)

        y = self.class_remapping(y)
        x = self._data_transformation(x, y, task_index)
        t = torch.ones(y.shape[0]) * task_index
        return x, y, t, task_index, np.arange(y.shape[0])
