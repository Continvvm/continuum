from typing import List, Tuple

import numpy as np

from clloader.datasets.base import _ContinuumDataset
from clloader.datasets.pytorch import (CIFAR10, CIFAR100, KMNIST, MNIST, FashionMNIST)


class Fellowship(_ContinuumDataset):

    def __init__(
        self,
        data_path: str = "",
        download: bool = True,
        dataset_list: List[_ContinuumDataset] = None
    ):
        super().__init__(data_path, download)

        self.datasets = [dataset(data_path, download) for dataset in dataset_list]

    def init(self, train: bool) -> Tuple[np.ndarray, np.ndarray]:
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
        super().__init__(data_path, download, dataset_list=[MNIST, FashionMNIST, KMNIST])


class CIFARFellowship(Fellowship):

    def __init__(self, data_path: str = "", download: bool = True) -> None:
        super().__init__(data_path, download, dataset_list=[CIFAR10, CIFAR100])
