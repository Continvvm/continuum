import abc
from typing import List, Tuple

import numpy as np
from torchvision import datasets as torchdata
from torchvision import transforms


class _ContinuumDataset(abc.ABC):

    def __init__(self, data_path: str = "", download: bool = True) -> None:
        self.data_path = data_path
        self.download = download

    @abc.abstractmethod
    def init(self, train: bool) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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


class PyTorchDataset(_ContinuumDataset):
    # TODO: some datasets have a different structure, like SVHN for ex. Handle it.
    def __init__(self, *args, dataset_type, **kwargs):
        super().__init__(*args, **kwargs)
        self.dataset_type = dataset_type

    def init(self, train: bool) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        dataset = self.dataset_type(self.data_path, download=self.download, train=train)
        x, y = np.array(dataset.data), np.array(dataset.targets)

        return x, y, None


class InMemoryDataset(_ContinuumDataset):

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

        self.train = (x_train, y_train, None)
        self.test = (x_test, y_test, None)
        self.is_path = is_path

    def init(self, train: bool) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if train:
            return self.train
        return self.test

    @property
    def in_memory(self) -> bool:
        return not self.is_path


class ImageFolderDataset(_ContinuumDataset):

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

    def init(self, train: bool) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if train:
            folder = self.train_folder
        else:
            folder = self.test_folder

        dataset = torchdata.ImageFolder(folder)
        return self._format(dataset.imgs)

    @staticmethod
    def _format(raw_data: List[Tuple[str, int]]) -> Tuple[np.ndarray, np.ndarray]:
        x = np.empty(len(raw_data), dtype="S255")
        y = np.empty(len(raw_data), dtype=np.int16)

        for i, (path, target) in enumerate(raw_data):
            x[i] = path
            y[i] = target

        return x, y, None
