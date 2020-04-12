import abc
from typing import List, Tuple, Union

import numpy as np

from torchvision import datasets as torchdata


class BaseDataset(abc.ABC):

    def __init__(self, data_path: str = "", download: bool = True) -> None:
        self.data_path = data_path
        self.download = download

    @abc.abstractmethod
    def init(self) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
        pass

    @property
    def class_order(self) -> List[int]:
        return [0]

    @property
    def in_memory(self):
        return True


class PyTorchDataset(BaseDataset):
    dataset_type = None

    def init(self) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
        train_dataset = self.dataset_type(self.data_path, download=self.download, train=True)
        x_train, y_train = train_dataset.data, np.array(train_dataset.targets)
        test_dataset = self.dataset_type(self.data_path, download=self.download, train=False)
        x_test, y_test = test_dataset.data, np.array(test_dataset.targets)

        return (x_train, y_train), (x_test, y_test)
