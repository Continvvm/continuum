import abc
import os
from typing import List, Tuple, Union

import numpy as np
from torchvision import datasets as torchdata
from torchvision import transforms


class BaseDataset(abc.ABC):

    def __init__(self, data_path: str = "", download: bool = True) -> None:
        self.data_path = data_path
        self.download = download

    @abc.abstractmethod
    def init(self) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
        pass

    @property
    def class_order(self) -> List[int]:
        return None

    @property
    def need_class_remapping(self) -> bool:
        """Flag for method `class_remapping`."""
        return False

    def class_remapping(self, class_ids: np.ndarray) -> np.ndarray:
        """Optional class remapping.

        Used for example in PermutedMNIST, cf transformed.py;

        :param class_ids: Original class_ids.
        :return: A remapping of the class ids.
        """
        return None

    @property
    def in_memory(self):
        return True

    @property
    def transformations(self):
        return [transforms.ToTensor()]


class PyTorchDataset(BaseDataset):
    # TODO: some datasets have a different structure, like SVHN for ex. Handle it.
    def __init__(self, *args, dataset_type, **kwargs):
        super().__init__(*args, **kwargs)
        self.dataset_type = dataset_type

    def init(self) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
        train_dataset = self.dataset_type(self.data_path, download=self.download, train=True)
        x_train, y_train = np.array(train_dataset.data), np.array(train_dataset.targets)
        test_dataset = self.dataset_type(self.data_path, download=self.download, train=False)
        x_test, y_test = np.array(test_dataset.data), np.array(test_dataset.targets)

        return (x_train, y_train), (x_test, y_test)


class InMemoryDataset(BaseDataset):

    def __init__(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_test: np.ndarray,
        y_test: np.ndarray,
        is_path: bool = False,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.train = (x_train, y_train)
        self.test = (x_test, y_test)
        self.is_path = is_path

    def init(self) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
        return self.train, self.test

    @property
    def in_memory(self) -> bool:
        return not self.is_path


class ImageFolderDataset(BaseDataset):

    def __init__(self, train_folder: str, test_folder: str, download: bool = True, **kwargs):
        super().__init__(download=download, **kwargs)

        self.train_folder = train_folder
        self.test_folder = test_folder

        if download:
            self._download()

    @property
    def in_memory(self):
        return False

    def _download(self):
        pass

    def init(self) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
        train_dataset = torchdata.ImageFolder(self.train_folder)
        train_data = self._format(train_dataset.imgs)

        test_dataset = torchdata.ImageFolder(self.test_folder)
        test_data = self._format(test_dataset.imgs)

        return train_data, test_data

    @staticmethod
    def _format(raw_data: List[Tuple[str, int]]) -> Tuple[np.ndarray, np.ndarray]:
        x = np.empty(len(raw_data), dtype="S255")
        y = np.empty(len(raw_data), dtype=np.int16)

        for i, (path, target) in enumerate(raw_data):
            x[i] = path
            y[i] = target

        return x, y
