import abc
from typing import List, Tuple, Union

import numpy as np

from clloader.datasets.base import BaseDataset, PyTorchDataset
from clloader.datasets.pytorch import (CIFAR10, CIFAR100, KMNIST, MNIST, FashionMNIST)
from torchvision import datasets as torchdata
from torchvision import transforms


class Fellowship(BaseDataset):

    def __init__(
        self, data_path: str = "", download: bool = True, dataset_list: List[BaseDataset] = None
    ):
        super().__init__(data_path, download)

        self.datasets = [dataset(data_path, download) for dataset in dataset_list]

    def init(self) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
        x_train, y_train = [], []
        x_test, y_test = [], []
        class_counter = 0

        for dataset in self.datasets:
            train, test = dataset.init()

            x_train.append(train[0])
            x_test.append(test[0])

            y_train.append(train[1] + class_counter)
            y_test.append(test[1] + class_counter)

            class_counter += len(np.unique(train[1]))

        x_train = np.concatenate(x_train)
        x_test = np.concatenate(x_test)
        y_train = np.concatenate(y_train)
        y_test = np.concatenate(y_test)

        return (x_train, y_train), (x_test, y_test)


class MNISTFellowship(Fellowship):

    def __init__(self, data_path: str = "", download: bool = True) -> None:
        super().__init__(data_path, download, dataset_list=[MNIST, FashionMNIST, KMNIST])


class CIFARFellowship(Fellowship):

    def __init__(self, data_path: str = "", download: bool = True) -> None:
        super().__init__(data_path, download, dataset_list=[CIFAR10, CIFAR100])
