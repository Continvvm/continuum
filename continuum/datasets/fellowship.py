from typing import List, Tuple, Union

import numpy as np

from continuum.datasets.base import _ContinuumDataset
from continuum.datasets.pytorch import (CIFAR10, CIFAR100, KMNIST, MNIST, FashionMNIST)


class Fellowship(_ContinuumDataset):
    """A dataset concatenating multiple datasets.

    :param datasets: A list of instanciated Continuum dataset.
    :param update_labels: if true, the class id are incremented for each new
                          concatenated datasets, otherwise the class ids are
                          shared, e.g. a Fellowship of CIFAR10+CIFAR100 with
                          update_labels=False will share the first 10 classes.
                          In doubt, let this param to True.
    :param proportions: Amount of data to take from each dataset. If int, take
                        that amount, if float take that percentage.
    :param seed: random seed to sample (if proportions is used) deterministically.
    """
    def __init__(
        self,
        datasets: List[_ContinuumDataset],
        update_labels: bool = True,
        proportions: Union[List[int], List[float], None] = None,
        seed: int = 1
    ):
        super().__init__()

        self.update_labels = update_labels
        self.datasets = datasets
        self.proportions = proportions
        self.seed = seed

    def get_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        x, y, t = [], [], []
        class_counter = 0

        for i, dataset in enumerate(self.datasets):
            x_, y_, _ = dataset.get_data()

            if self.update_labels:
                y_ = y_ + class_counter
            else:
                y_ = y_

            if self.proportions:
                indexes = _balanced_sampling(_y, self.proportions[i], self.seed)
                x_ = x_[indexes]
                y_ = y_[indexes]

            x.append(x_)
            y.append(y_)
            t.append(np.ones(len(_x)) * i)

            class_counter += len(np.unique(data[1]))

        x = np.concatenate(x)
        y = np.concatenate(y)
        t = np.concatenate(t)

        # There should be by default as much task id as datasets
        assert len(np.unique(t)) == len(self.datasets), f'They should be as much datasets as task ids,' \
                                                        f' we have {len(self.datasets)} datasets vs' \
                                                        f' {len(np.unique(t))} task ids'

        return x, y, t


def _balanced_sampling(y, amount, seed):
    if isinstance(amount, float):
        amount = int(len(y) * amount)
    unique_classes = np.unique(y)
    if len(unique_classes) > amount:
        raise ValueError(
            f"Not enough amount ({amount}) for the number of classes ({len(unique_classes)})."
        )
    # There can be a few images lost, but that's not very important.
    amount_per_class = int(amount / unique_classes)

    rng = np.random.RandomState(seed=seed)

    indexes = []
    for c in unique_classes:
        class_indexes = np.where(y == c)
        indexes.append(
            rng.choice(class_indexes, size=amount_per_class)
        )

    return np.concatenate(indexes)


class MNISTFellowship(Fellowship):

    def __init__(self,
                 data_path: str = "",
                 train: bool = True,
                 download: bool = True,
                 update_labels: bool = True) -> None:
        super().__init__(
            datasets=[
                MNIST(data_path=data_path, train=train, download=download),
                FashionMNIST(data_path=data_path, train=train, download=download),
                KMNIST(data_path=data_path, train=train, download=download)
            ],
            update_labels=update_labels
        )


class CIFARFellowship(Fellowship):

    def __init__(self,
                 data_path: str = "",
                 train: bool = True,
                 download: bool = True,
                 update_labels: bool = True) -> None:
        super().__init__(
            datasets=[
                CIFAR10(data_path=data_path, train=train, download=download),
                CIFAR100(data_path=data_path, train=train, download=download)
            ],
            update_labels=update_labels
        )
