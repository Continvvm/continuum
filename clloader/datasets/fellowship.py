from typing import List, Tuple, Type

import numpy as np

from clloader.datasets.base import _ContinuumDataset
from clloader.datasets.pytorch import (CIFAR10, CIFAR100, KMNIST, MNIST, FashionMNIST)


class Fellowship(_ContinuumDataset):

    def __init__(
        self,
        dataset_list: List[Type[_ContinuumDataset]],
        data_path: str = "",
        download: bool = True,
    ):
        super().__init__(data_path, download)

        self.datasets = [dataset(data_path, download) for dataset in dataset_list]

    def init(self, train: bool) -> Tuple[np.ndarray, np.ndarray, None]:
        x, y = [], []
        class_counter = 0

        for dataset in self.datasets:
            data = dataset.init(train)

            x.append(data[0])
            y.append(data[1] + class_counter)

            class_counter += len(np.unique(data[1]))

        x = np.concatenate(x)
        y = np.concatenate(y)

        return x, y, None


class MNISTFellowship(Fellowship):

    def __init__(self, data_path: str = "", download: bool = True) -> None:
        super().__init__([MNIST, FashionMNIST, KMNIST], data_path, download)


class CIFARFellowship(Fellowship):

    def __init__(self, data_path: str = "", download: bool = True) -> None:
        super().__init__([CIFAR10, CIFAR100], data_path, download)
