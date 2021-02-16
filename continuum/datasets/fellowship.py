from typing import List, Tuple, Type

import numpy as np

from continuum.datasets.base import _ContinuumDataset
from continuum.datasets.pytorch import (CIFAR10, CIFAR100, KMNIST, MNIST, FashionMNIST)


class Fellowship(_ContinuumDataset):

    def __init__(
        self,
        dataset_list: List[Type[_ContinuumDataset]],
        data_path: str = "",
        train: bool = True,
        download: bool = True,
        update_labels: bool = True,
    ):
        super().__init__(data_path, download)

        self.update_labels = update_labels
        self.datasets = [
            dataset(data_path=data_path, train=train, download=download) for dataset in dataset_list
        ]


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
            dataset_list=[MNIST, FashionMNIST, KMNIST],
            train=train,
            data_path=data_path,
            download=download,
            update_labels=update_labels
        )


class CIFARFellowship(Fellowship):

    def __init__(self,
                 data_path: str = "",
                 train: bool = True,
                 download: bool = True,
                 update_labels: bool = True) -> None:
        super().__init__(
            dataset_list=[CIFAR10, CIFAR100],
            train=train,
            data_path=data_path,
            download=download,
            update_labels=update_labels
        )
