from typing import List, Tuple

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
    """
    def __init__(
        self,
        datasets: List[_ContinuumDataset],
        update_labels: bool = True,
    ):
        super().__init__()

        self.update_labels = update_labels
        self.datasets = datasets


    def get_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        x, y, t = [], [], []
        class_counter = 0

        for i, dataset in enumerate(self.datasets):
            data = dataset.get_data()

            x.append(data[0])
            if self.update_labels:
                y.append(data[1] + class_counter)
            else:
                y.append(data[1])

            t.append(np.ones(len(data[0])) * i)

            class_counter += len(np.unique(data[1]))

        x = np.concatenate(x)
        y = np.concatenate(y)
        t = np.concatenate(t)

        # There should be by default as much task id as datasets
        assert len(np.unique(t)) == len(self.datasets), f'They should be as much datasets as task ids,' \
                                                        f' we have {len(self.datasets)} datasets vs' \
                                                        f' {len(np.unique(t))} task ids'

        return x, y, t


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
