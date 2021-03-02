from typing import Tuple
from continuum.tasks import TaskSet

import numpy as np


def split_train_val(dataset: TaskSet, val_split: float = 0.1) -> Tuple[TaskSet, TaskSet]:
    """Split train dataset into two datasets, one for training and one for validation.

    :param dataset: A torch dataset, with .x and .y attributes.
    :param val_split: Percentage to allocate for validation, between [0, 1[.
    :return: A tuple a dataset, respectively for train and validation.
    """
    random_state = np.random.RandomState(seed=1)
    indexes = np.arange(len(dataset))
    random_state.shuffle(indexes)

    train_indexes = indexes[int(val_split * len(indexes)):]
    val_indexes = indexes[:int(val_split * len(indexes))]

    x_train, y_train, t_train = dataset.get_raw_samples(train_indexes)
    train_dataset = TaskSet(x_train, y_train, t_train, dataset.trsf, dataset.data_type)

    x_val, y_val, t_val = dataset.get_raw_samples(val_indexes)
    val_dataset = TaskSet(x_val, y_val, t_val, dataset.trsf, dataset.data_type)

    return train_dataset, val_dataset


def concat(dataset: TaskSet, *others: TaskSet) -> TaskSet:
    """Concatenate a dataset A with one or many *other* datasets.

    The transformations will be those of the first dataset.

    :param dataset: A task set.
    :param others: One or many others task sets.
    :return: A concatenated task set.
    """
    x, y, t = [dataset._x], [dataset._y], [dataset._t]

    for task_set in others:
        if task_set.data_type != dataset.data_type:
            raise Exception(
                f"Invalid data type {task_set.data_type} != {dataset.data_type}"
            )

        x.append(task_set._x)
        y.append(task_set._y)
        t.append(task_set._t)

    return TaskSet(
        np.concatenate(x),
        np.concatenate(y),
        np.concatenate(t),
        trsf=dataset.trsf,
        data_type=dataset.data_type
    )
