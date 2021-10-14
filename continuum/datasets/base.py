import abc
import os
from typing import List, Tuple, Union
import warnings

import numpy as np
from torchvision import datasets as torchdata
from torchvision import transforms

from continuum.transforms.segmentation import ToTensor as ToTensorSegmentation
from continuum.tasks import TaskType


class _ContinuumDataset(abc.ABC):

    def __init__(self, data_path: str = "", train: bool = True, download: bool = True) -> None:
        self.data_path = os.path.expanduser(data_path)
        self.download = download
        self.train = train

        if self.data_path is not None and self.data_path != "" and not os.path.exists(self.data_path):
            os.makedirs(self.data_path)

        if self.download:
            self._download()

        self.number_classes = None

        if not isinstance(self.data_type, TaskType):
            raise NotImplementedError(
                f"Dataset's data_type ({self.data_type}) is not supported."
                " It must be a member of the enum TaskType."
            )

    def get_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        pass

    @property
    def classes(self) -> List:
        """Return list of classes in the dataset"""
        pass

    @property
    def nb_classes(self) -> int:
        """Return number of classes in the dataset"""
        pass

    def _download(self):
        pass

    @property
    def class_order(self) -> Union[None, List[int]]:
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
        return class_ids

    @property
    def data_type(self) -> TaskType:
        return TaskType.IMAGE_ARRAY

    @property
    def transformations(self):
        """Default transformations if nothing is provided to the scenario."""
        if self.data_type == TaskType.SEGMENTATION:
            return [ToTensorSegmentation()]
        return [transforms.ToTensor()]

    @property
    def bounding_boxes(self) -> List:
        """Returns a bounding box (x1, y1, x2, y2) per sample if they need to be cropped."""
        return None

    @property
    def attributes(self) -> np.ndarray:
        """Returns normalized attributes for all class if available.

        Those attributes can often be found in dataset used for Zeroshot such as
        CUB200, or AwA. The matrix shape is (nb_classes, nb_attributes), and it
        has been L2 normalized along side its attributes dimension.
        """
        return None

    @property
    def classes(self) -> List:
        """Return list of classes in the dataset"""
        return list(np.arange(self.nb_classes))

    @property
    def nb_classes(self) -> int:
        """Return number of classes in the dataset"""
        return self.number_classes


class _SemanticSegmentationDataset(_ContinuumDataset):
    """Base class for segmentation-based dataset."""

    @property
    def data_type(self) -> str:
        return TaskType.SEGMENTATION

class PyTorchDataset(_ContinuumDataset):
    """Continuum version of torchvision datasets.
    :param dataset_type: A Torchvision dataset, like MNIST or CIFAR100.
    :param train: train flag
    :param download: download
    """

    # TODO: some datasets have a different structure, like SVHN for ex. Handle it.
    def __init__(
            self, data_path: str = "", dataset_type=None, train: bool = True, download: bool = True, **kwargs):

        if "transform" in kwargs:
            raise ValueError(
                "Don't provide `transform` to the dataset. "
                "You should give those to the scenario."
            )

        super().__init__(data_path=data_path, train=train, download=download)

        self.dataset_type = dataset_type
        self.dataset = self.dataset_type(self.data_path, download=self.download, train=self.train, **kwargs)

    def get_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        x, y = np.array(self.dataset.data), np.array(self.dataset.targets)

        if 0 not in y:
            # This case can happen when the first class id is 1 and not 0.
            # For example in EMNIST with 'letters' split (WTF right).
            # TODO: We should handle this case in a more generic fashion later.
            warnings.warn("Converting 1-based class ids to 0-based class ids.")
            y -= 1

        return x, y, None

    @property
    def classes(self) -> List:
        """Return list of classes in the dataset"""
        return list(np.unique(self.dataset.targets))

    @property
    def nb_classes(self) -> int:
        """Return number of classes in the dataset"""
        return len(self.classes)


class InMemoryDataset(_ContinuumDataset):
    """Continuum dataset for in-memory data.

    :param x_train: Numpy array of images or paths to images for the train set.
    :param y_train: Targets for the train set.
    :param data_type: Format of the data.
    :param t_train: Optional task ids for the train set.
    """

    def __init__(
            self,
            x: np.ndarray,
            y: np.ndarray,
            t: Union[None, np.ndarray] = None,
            data_type: TaskType = TaskType.IMAGE_ARRAY,
            train: bool = True,
            download: bool = True,
    ):
        self._data_type = data_type
        super().__init__(train=train, download=download)

        if len(x) != len(y):
            raise ValueError(f"Number of datapoints ({len(x)}) != number of labels ({len(y)})!")
        if t is not None and len(t) != len(x):
            raise ValueError(f"Number of datapoints ({len(x)}) != number of task ids ({len(t)})!")

        self.data = (x, y, t)

    def get_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self.data

    @property
    def data_type(self) -> TaskType:
        return self._data_type

    @data_type.setter
    def data_type(self, data_type: TaskType) -> None:
        self._data_type = data_type

    @property
    def classes(self) -> List:
        """Return list of classes in the dataset"""
        return list(np.unique(self.data[1]))

    @property
    def nb_classes(self) -> int:
        """Return number of classes in the dataset"""
        return len(self.classes)

class ImageFolderDataset(_ContinuumDataset):
    """Continuum dataset for datasets with tree-like structure.

    :param train_folder: The folder of the train data.
    :param test_folder: The folder of the test data.
    :param download: Dummy parameter.
    """

    def __init__(
        self,
        data_path: str,
        train: bool = True,
        download: bool = True,
        data_type: TaskType = TaskType.IMAGE_PATH
    ):
        self.data_path = data_path
        self._data_type = data_type
        super().__init__(data_path=data_path, train=train, download=download)

        allowed_data_types = (TaskType.IMAGE_PATH, TaskType.SEGMENTATION)
        if data_type not in allowed_data_types:
            raise ValueError(f"Invalid data_type={data_type}, allowed={allowed_data_types}.")

        self.number_classes = None

    @property
    def data_type(self) -> TaskType:
        return self._data_type

    def get_data(self) -> Tuple[np.ndarray, np.ndarray, Union[None, np.ndarray]]:
        self.dataset = torchdata.ImageFolder(self.data_path)
        x, y, t = self._format(self.dataset.imgs)
        self.list_classes = np.unique(y)
        return x, y, t

    @staticmethod
    def _format(raw_data: List[Tuple[str, int]]) -> Tuple[np.ndarray, np.ndarray, None]:
        x = np.empty(len(raw_data), dtype="S255")
        y = np.empty(len(raw_data), dtype=np.int16)

        for i, (path, target) in enumerate(raw_data):
            x[i] = path
            y[i] = target

        return x, y, None


    @property
    def classes(self) -> List:
        """Return list of classes in the dataset"""
        return list(np.arange(self.nb_classes))

    @property
    def nb_classes(self) -> int:
        """Number of classes"""
        if self.number_classes is None:
            self.dataset = torchdata.ImageFolder(self.data_path)
            x, y, t = self._format(self.dataset.imgs)
            self.number_classes = len(np.unique(y))
        return self.number_classes
