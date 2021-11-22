import abc
import os
import warnings
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
import h5py
from torchvision import datasets as torchdata
from torchvision import transforms

from continuum.tasks import TaskSet, TaskType
from continuum.transforms import segmentation as transforms_seg
from continuum import utils


class _ContinuumDataset(abc.ABC):

    def __init__(self, data_path: str = "", train: bool = True, download: bool = True) -> None:
        self.data_path = os.path.expanduser(data_path) if data_path is not None else None
        self.download = download
        self.train = train

        if self.data_path is not None and self.data_path != "" and not os.path.exists(self.data_path):
            os.makedirs(self.data_path)

        if self.download:
            self._download()

        if not isinstance(self.data_type, TaskType):
            raise NotImplementedError(
                f"Dataset's data_type ({self.data_type}) is not supported."
                " It must be a member of the enum TaskType."
            )

        # Initialization of the default properties
        if self.data_type == TaskType.SEGMENTATION:
            self._trsf = [transforms_seg.ToTensor()]
        else:
            self._trsf = [transforms.ToTensor()]
        self._bboxes = None
        self._attributes = None

    def get_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Returns the loaded data under the form of x, y, and t."""
        raise NotImplementedError("This method should be implemented!")

    def _download(self):
        pass

    def slice(
            self,
            keep_classes: Optional[List[int]] = None,
            discard_classes: Optional[List[int]] = None,
            keep_tasks: Optional[List[int]] = None,
            discard_tasks: Optional[List[int]] = None
    ):
        """Slice dataset to keep/discard some classes/task-ids.

        Note that keep_* and and discard_* are mutually exclusive.
        Note also that if a selection (keep or discard) is being made on the classes
        and on the task ids, the resulting intersection will be taken.

        :param keep_classes: Only keep samples with these classes.
        :param discard_classes: Discard samples with these classes.
        :param keep_tasks: Only keep samples with these task ids.
        :param discard_tasks: Discard samples with these task ids.
        :return: A new Continuum dataset ready to be given to a scenario.
        """
        if self.data_type == TaskType.SEGMENTATION:
            raise NotImplementedError("It's not possible yet to slice Segmentation datasets.")

        x, y, t = self.get_data()

        indexes = utils._slice(
            y, t,
            keep_classes, discard_classes,
            keep_tasks, discard_tasks
        )

        new_x, new_y, new_t = x[indexes], y[indexes], t[indexes]
        sliced_dataset = InMemoryDataset(
            new_x, new_y, new_t,
            data_type=self.data_type
        )
        sliced_dataset.attributes = self.attributes
        sliced_dataset.bounding_boxes = self.bounding_boxes
        sliced_dataset.transformations = self.transformations

        return sliced_dataset

    @property
    def nb_classes(self) -> List[int]:
        return None

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

    def to_taskset(
            self,
            trsf: Optional[List[Callable]] = None,
            target_trsf: Optional[List[Callable]] = None
    ) -> TaskSet:
        """Returns a TaskSet that can be directly given to a torch's DataLoader.

        You can use this method if you don't care about the continual aspect and
        simply want to use the datasets in a classical supervised setting.

        :param trsf: List of transformations to be applied on x.
        :param target_trsf: List of transformations to be applied on y.
        :return taskset: A taskset which implement the interface of torch's Dataset.
        """
        if trsf is None and self.data_type == TaskType.SEGMENTATION:
            trsf = transforms_seg.Compose(self.transformations)
        elif trsf is None:
            trsf = transforms.Compose(self.transformations)

        return TaskSet(
            *self.get_data(),
            trsf=trsf,
            target_trsf=target_trsf,
            data_type=self.data_type,
            bounding_boxes=self.bounding_boxes
        )

    @property
    def class_order(self) -> Union[None, List[int]]:
        return None

    @property
    def need_class_remapping(self) -> bool:
        """Flag for method `class_remapping`."""
        return False

    @property
    def data_type(self) -> TaskType:
        return TaskType.IMAGE_ARRAY

    @property
    def transformations(self):
        """Default transformations if nothing is provided to the scenario."""
        return self._trsf

    @transformations.setter
    def transformations(self, trsf: List[Callable]):
        self._trsf = trsf

    @property
    def bounding_boxes(self) -> List:
        """Returns a bounding box (x1, y1, x2, y2) per sample if they need to be cropped."""
        return self._bboxes

    @bounding_boxes.setter
    def bounding_boxes(self, bboxes: List):
        self._bboxes = bboxes

    @property
    def attributes(self) -> np.ndarray:
        """Returns normalized attributes for all class if available.

        Those attributes can often be found in dataset used for Zeroshot such as
        CUB200, or AwA. The matrix shape is (nb_classes, nb_attributes), and it
        has been L2 normalized along side its attributes dimension.
        """
        return self._attributes

    @attributes.setter
    def attributes(self, attributes: np.ndarray):
        self._attributes = attributes


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
        self._nb_classes = len(np.unique(y))

    def get_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self.data

    @property
    def nb_classes(self) -> List[int]:
        return self._nb_classes

    @property
    def data_type(self) -> TaskType:
        return self._data_type

    @data_type.setter
    def data_type(self, data_type: TaskType) -> None:
        self._data_type = data_type


class H5Dataset(_ContinuumDataset):
    """Continuum dataset for in-memory data with h5 file.
    This class either creates a h5 dataset or reload an existing one.

    :param x_train: Numpy array of images or paths to images for the train set.
    :param y_train: Targets for the train set.
    :param data_type: Format of the data.
    :param t_train: Optional task ids for the train set.
    """

    def __init__(
            self,
            x: np.ndarray = None,
            y: np.ndarray = None,
            t: Union[None, np.ndarray] = None,
            data_path: str = "h5_dataset.h5",
            train: bool = True,
            download: bool = True,
    ):
        self._data_type = TaskType.H5
        super().__init__(data_path=None, train=train, download=download)

        if x is None:
            # Load an existing h5_dataset
            self._check_existing_file(data_path)
        else:
            if len(x) != len(y):
                raise ValueError(f"Number of datapoints ({len(x)}) != number of labels ({len(y)})!")
            self.no_task_index = False
            if t is None:
                self.no_task_index = True
            else:
                if len(t) != len(x):
                    raise ValueError(f"Number of datapoints ({len(x)}) != number of task ids ({len(t)})!")

        self.data_path = data_path

        if x is not None:
            self.create_file(x, y, t, self.data_path)

    @property
    def data_type(self) -> TaskType:
        return TaskType.H5

    def __len__(self):
        return len(self.get_class_vector())

    def _check_existing_file(self, filename):
        if not os.path.exists(filename):
            raise IOError(f"You can not load unexisting file : {filename}")

        with h5py.File(filename, 'r') as hf:
            data_vector = hf['x'][:]
            classes_vector = hf['y'][:]
            if 't' in hf.keys():
                self.no_task_index = False
                task_index_vector = hf['t'][:]
                if task_index_vector is None:
                    self.no_task_index = True
            else:
                self.no_task_index = True

            assert len(classes_vector) == len(data_vector)
            if not self.no_task_index:
                assert len(classes_vector) == len(task_index_vector)

        self.data_path = filename

    def slice(
            self,
            new_h5_path: str,
            keep_classes: Optional[List[int]] = None,
            discard_classes: Optional[List[int]] = None,
            keep_tasks: Optional[List[int]] = None,
            discard_tasks: Optional[List[int]] = None
    ):
        """Slice dataset to keep/discard some classes/task-ids.

        Note that keep_* and and discard_* are mutually exclusive.
        Note also that if a selection (keep or discard) is being made on the classes
        and on the task ids, the resulting intersection will be taken.

        :param new_h5_path: A path where to store the sliced dataset as H5.
        :param keep_classes: Only keep samples with these classes.
        :param discard_classes: Discard samples with these classes.
        :param keep_tasks: Only keep samples with these task ids.
        :param discard_tasks: Discard samples with these task ids.
        :return: A new Continuum dataset ready to be given to a scenario.
        """
        _, y, t = self.get_data()

        indexes = utils._slice(
            y, t,
            keep_classes, discard_classes,
            keep_tasks, discard_tasks
        )

        with h5py.File(self.data_path, 'r') as hf:
            new_x = hf['x'][indexes]

        new_y, new_t = y[indexes], t[indexes]
        sliced_dataset = H5Dataset(
            new_x, new_y, new_t,
            data_path=new_h5_path
        )

        return sliced_dataset

    def create_file(self, x, y, t, data_path):
        """"Create and initiate h5 file with data, labels and task index (if not none)"""

        assert not os.path.exists(data_path), print(f"You can not replace file : {data_path}")

        with h5py.File(data_path, 'w') as hf:
            hf.create_dataset('x', data=x, chunks=True, maxshape=([None] + list(x[0].shape)))
            hf.create_dataset('y', data=y, chunks=True, maxshape=([None]))
            if not self.no_task_index:
                hf.create_dataset('t', data=t, chunks=True, maxshape=([None]))

    def get_task_indexes(self):
        """"Return the whole vector of task index"""
        task_indexe_vector = None
        if not self.no_task_index:
            with h5py.File(self.data_path, 'r') as hf:
                task_indexe_vector = hf['t'][:]
        return task_indexe_vector

    def get_task_index(self, index):
        """"Return one task index value value for a given index"""
        task_indexes_value = None
        if not self.no_task_index:
            with h5py.File(self.data_path, 'r') as hf:
                task_indexes_value = hf['t'][index]
        return task_indexes_value

    def get_class_vector(self):
        """"Return the whole vector of classes"""
        classes_vector = None
        with h5py.File(self.data_path, 'r') as hf:
            classes_vector = hf['y'][:]
        return classes_vector

    def get_class(self, index):
        """"Return one class value for a given index"""
        class_value = None
        with h5py.File(self.data_path, 'r') as hf:
            class_value = hf['y'][index]
        return class_value

    def add_data(self, x, y, t):
        """"This method is here to be able to build the h5 by part"""
        if not (self.no_task_index == (t is None)):
            raise AssertionError("You can not add data with task index to h5 without task index or the opposite")

        with h5py.File(self.data_path, 'a') as hf:
            reshape_size = hf["t"].shape[0] + t.shape[0]
            hf['x'].resize(reshape_size, axis=0)
            hf["x"][-x.shape[0]:] = x
            hf['y'].resize(reshape_size, axis=0)
            hf["y"][-x.shape[0]:] = y
            if not self.no_task_index:
                hf['t'].resize(reshape_size, axis=0)
                hf["t"][-x.shape[0]:] = t

    def get_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self.data_path, self.get_class_vector(), self.get_task_indexes()


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
