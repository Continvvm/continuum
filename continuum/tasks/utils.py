from typing import Tuple, List

import torch
import numpy as np

from continuum.tasks.base import BaseTaskSet
from continuum.tasks.task_set import TaskSet, TaskType
import torch
import numpy as np

from continuum.tasks.base import BaseTaskSet, TaskType
from continuum.tasks.task_set import TaskSet


def get_balanced_sampler(taskset, log=False):
    """Create a sampler that will balance the dataset.

    You should give the returned sampler to the dataloader with the argument `sampler`.

    :param taskset: A pytorch dataset that implement the TaskSet interface.
    :param log: Use a log weights. If enabled, there will still be imbalance but
                on the other hand, the oversampling/downsampling won't be as violent.
    :return: A PyTorch sampler.
    """
    if taskset.data_type in (TaskType.SEGMENTATION, TaskType.OBJ_DETECTION, TaskType.TEXT):
        raise NotImplementedError(
            "Samplers are not yet available for the "
            f"{taskset.data_type} type."
        )

    y = taskset.get_raw_samples()[1]
    nb_per_class = np.bincount(y)
    weights_per_class = 1 / nb_per_class
    if log:
        weights_per_class = np.log(weights_per_class)
        weights_per_class = 1 - (weights_per_class / np.sum(weights_per_class))

    weights = weights_per_class[y]

    return torch.utils.data.sampler.WeightedRandomSampler(weights, len(taskset))


def split_train_val(dataset: BaseTaskSet, val_split: float = 0.1) -> Tuple[BaseTaskSet, BaseTaskSet]:
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

    print(train_indexes)
    print(val_indexes)

    if dataset.data_type != TaskType.H5:
        x_train, y_train, t_train = dataset.get_raw_samples(train_indexes)
        x_val, y_val, t_val = dataset.get_raw_samples(val_indexes)
        idx_train, idx_val = None, None
    else:
        y_train, y_val, t_train, t_val = None, None, None, None
        if dataset._y is not None:
            y_train = dataset._y[train_indexes]
            y_val = dataset._y[val_indexes]

        if dataset._t is not None:
            t_train = dataset._t[train_indexes]
            t_val = dataset._t[val_indexes]
        idx_train = dataset.data_indexes[train_indexes]
        idx_val = dataset.data_indexes[val_indexes]

        x_train = dataset.h5_filename
        x_val = dataset.h5_filename

    train_dataset = TaskSet(x_train, y_train, t_train,
                            trsf=dataset.trsf,
                            data_type=dataset.data_type,
                            data_indexes=idx_train)
    val_dataset = TaskSet(x_val, y_val, t_val,
                          trsf=dataset.trsf,
                          data_type=dataset.data_type,
                          data_indexes=idx_val)

    return train_dataset, val_dataset


def concat(task_sets: List[BaseTaskSet]) -> BaseTaskSet:
    """Concatenate a dataset A with one or many *other* datasets.

    The transformations will be those of the first dataset.

    :param task_sets: A list of task sets.
    :return: A concatenated task set.
    """
    x, y, t = [], [], []

    data_type = task_sets[0].data_type

    for task_set in task_sets:
        if task_set.data_type != data_type:
            raise Exception(
                f"Invalid data type {task_set.data_type} != {data_type}"
            )

        x.append(task_set._x)
        y.append(task_set._y)
        t.append(task_set._t)

    return TaskSet(
        np.concatenate(x),
        np.concatenate(y),
        np.concatenate(t),
        trsf=task_sets[0].trsf,
        data_type=data_type
    )
