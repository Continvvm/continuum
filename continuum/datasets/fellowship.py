from typing import List, Tuple, Type

import numpy as np

from continuum.datasets.base import _ContinuumDataset
from continuum.datasets.pytorch import (CIFAR10, CIFAR100, KMNIST, MNIST, FashionMNIST)


class Fellowship(_ContinuumDataset):

    def __init__(
        self,
        dataset_list: List[Type[_ContinuumDataset]],
        data_path: str = "",
        download: bool = True,
    ):
        super(Fellowship, self).__init__(data_path, download)

        self.datasets = [dataset(data_path, download) for dataset in dataset_list]


    def get_data(self) -> Tuple[np.ndarray, np.ndarray,  np.ndarray]:
        x, y, t = [], [], []
        class_counter = 0

        for i, dataset in enumerate(self.datasets):
            data = dataset.get_data()

            x.append(data[0])
            y.append(data[1] + class_counter)
            t.append(np.ones(len(data[0])*i))

            class_counter += len(np.unique(data[1]))

        x = np.concatenate(x)
        y = np.concatenate(y)
        t = np.concatenate(t)

        return x, y, t


class MNISTFellowship(Fellowship):

    def __init__(self, data_path: str = "", download: bool = True) -> None:
        super(MNISTFellowship, self).__init__([MNIST, FashionMNIST, KMNIST], data_path, download)


class CIFARFellowship(Fellowship):

    def __init__(self, data_path: str = "", download: bool = True) -> None:
        super(CIFARFellowship, self).__init__([CIFAR10, CIFAR100], data_path, download)
