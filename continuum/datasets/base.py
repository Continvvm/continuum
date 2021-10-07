import abc
import os
from typing import List, Tuple, Union
import warnings

import numpy as np
import h5py
from torchvision import datasets as torchdata
from torchvision import transforms

from continuum.transforms.segmentation import ToTensor as ToTensorSegmentation
from continuum.tasks import TaskType


class _ContinuumDataset(abc.ABC):

    def __init__(self, data_path: str = "", train: bool = True, download: bool = True) -> None:
        self.data_path = os.path.expanduser(data_path)
        self.download = download
        self.train = train
        self.dataset_type = "array"

        if self.data_path is not None and self.data_path != "" and not os.path.exists(self.data_path):
            os.makedirs(self.data_path)

        if self.download:
            self._download()

        if not isinstance(self.data_type, TaskType):
            raise NotImplementedError(
                f"Dataset's data_type ({self.data_type}) is not supported."
                " It must be a member of the enum TaskType."
            )

    def get_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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


class H5Dataset(InMemoryDataset):
    """Continuum dataset for in-memory data with h5 file. There are suppose to be compatible only with ContinualScenario

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
            data_path: str = "",
            data_type: TaskType = TaskType.IMAGE_ARRAY,
            train: bool = True,
            download: bool = True,
    ):
        super().__init__(x, y, t, data_type, train=train, download=download)
        self._data_type = data_type
        self.data_path = data_path
        self.dataset_type = "h5"

        assert t is not None, AssertionError("This dataset is made for predefined t vector")

        if len(x) != len(y):
            raise ValueError(f"Number of datapoints ({len(x)}) != number of labels ({len(y)})!")
        if len(t) != len(x):
            raise ValueError(f"Number of datapoints ({len(x)}) != number of task ids ({len(t)})!")

        self.create_file(x, y, t, data_path)

    def create_file(self, x, y, t, data_path):

        with h5py.File(data_path, 'w') as hf:
            task_indexes = np.unique(t)
            for task_index in task_indexes:
                data_indexes = np.where(t == task_index)[0]

                g1 = hf.create_group(f'task-{task_index}')
                g1.create_dataset('x', data=x[data_indexes])
                g1.create_dataset('y', data=y[data_indexes])
                g1.create_dataset('t', data=t[data_indexes])

    def keys(self):
        with h5py.File(self.data_path, 'r') as hf:
            keys = list(hf.keys())
        return keys

    def add_data(self, x, y, t):
        """"This method is here to be able to build the h5 by part"""

        with h5py.File(self.data_path, 'w') as hf:
            task_indexes = np.unique(t)
            for task_index in task_indexes:
                data_indexes = np.where(t == task_index)[0]

                if f'task-{task_index}' in hf.keys():
                    self._update(hf, x, y, t, str_key=f"task-{task_index}")
                else:
                    g1 = hf.create_group(f'task-{task_index}')
                    g1.create_dataset('x', data=x[data_indexes])
                    g1.create_dataset('y', data=y[data_indexes])
                    g1.create_dataset('t', data=t[data_indexes])

    def _update(self, file_hf, x, y, t, str_key):

        h5x = file_hf.get(f'{str_key}/x')
        h5y = file_hf.get(f'{str_key}/y')
        h5t = file_hf.get(f'{str_key}/t')
        del file_hf[str_key]
        x = np.concatenate([h5x, x], axis=0)
        y = np.concatenate([h5y, y], axis=0)
        t = np.concatenate([h5t, t], axis=0)
        g1 = file_hf.create_group(str_key)
        g1.create_dataset('x', data=x)
        g1.create_dataset('y', data=y)
        g1.create_dataset('t', data=t)

    def get_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        raise AssertionError("h5dataset are not made to load all data in one time. Use get_task_data instead.")

    def get_task_data(self, ind_task: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        task_data = None
        with h5py.File(self.data_path, 'r') as hf:
            hf = h5py.File(self.data_path, 'r')
            task_data = hf.get(f'task-{ind_task}')
        return [task_data['x'], task_data['y'], task_data['t']]


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

    @property
    def data_type(self) -> TaskType:
        return self._data_type

    def get_data(self) -> Tuple[np.ndarray, np.ndarray, Union[None, np.ndarray]]:
        self.dataset = torchdata.ImageFolder(self.data_path)
        return self._format(self.dataset.imgs)

    @staticmethod
    def _format(raw_data: List[Tuple[str, int]]) -> Tuple[np.ndarray, np.ndarray, None]:
        x = np.empty(len(raw_data), dtype="S255")
        y = np.empty(len(raw_data), dtype=np.int16)

        for i, (path, target) in enumerate(raw_data):
            x[i] = path
            y[i] = target

        return x, y, None
