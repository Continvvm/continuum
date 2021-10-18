import enum
from copy import copy
from typing import Tuple, Union, Optional, List

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset as TorchDataset
from torchvision import transforms


class TaskType(enum.Enum):
    """Enumeration to list all possible data types supported."""
    IMAGE_ARRAY = 1
    IMAGE_PATH = 2
    TEXT = 3
    TENSOR = 4
    SEGMENTATION = 5
    OBJ_DETECTION = 6
    H5 = 7


def _tensorize_list(x):
    if isinstance(x[0], torch.Tensor):
        return torch.stack(x)
    return torch.tensor(x)


class BaseTaskSet(TorchDataset):
    """A task dataset returned by the CLLoader.

    :param x: The data, either image-arrays or paths to images saved on disk.
    :param y: The targets, not one-hot encoded.
    :param t: The task id of each sample.
    :param trsf: The transformations to apply on the images.
    :param data_type: Type of the data, either "image_path", "image_array",
                      "text", "tensor" or "segmentation".
    """

    def __init__(
            self,
            x: np.ndarray,
            y: np.ndarray,
            t: np.ndarray,
            trsf: Union[transforms.Compose, List[transforms.Compose]],
            target_trsf: Optional[Union[transforms.Compose, List[transforms.Compose]]] = None,
            bounding_boxes: Optional[np.ndarray] = None
    ):
        self._x, self._y, self._t = x, y, t

        # if task index are not provided t is always -1
        if self._t is None:
            self._t = -1 * np.ones_like(y, dtype=np.int64)

        self.trsf = trsf
        self.target_trsf = target_trsf
        self.data_type = TaskType.TENSOR
        self.bounding_boxes = bounding_boxes
        self.bounding_boxes = bounding_boxes

        self._to_tensor = transforms.ToTensor()

    @property
    def nb_classes(self):
        """The number of classes contained in the current task."""
        return len(np.unique(self._y))

    def get_classes(self):
        """Array of all classes contained in the current task."""
        return np.unique(self._y)

    def concat(self, *task_sets):
        """Concat others task sets.

        :param task_sets: One or many task sets.
        """
        for task_set in task_sets:
            if task_set.data_type != self.data_type:
                raise Exception(
                    f"Invalid data type {task_set.data_type} != {self.data_type}, "
                    "all concatenated tasksets must be of the same type."
                )

            self.add_samples(task_set._x, task_set._y, task_set._t)

    def add_samples(self, x: np.ndarray, y: np.ndarray, t: Union[None, np.ndarray] = None):
        """Add memory for rehearsal.

        :param x: Sampled data chosen for rehearsal.
        :param y: The associated targets of `x_memory`.
        :param t: The associated task ids. If not provided, they will be
                         defaulted to -1.
        """
        self._x = np.concatenate((self._x, x))
        self._y = np.concatenate((self._y, y))
        if t is not None:
            self._t = np.concatenate((self._t, t))
        else:
            self._t = np.concatenate((self._t, -1 * np.ones(len(x))))

    def plot(
            self,
            path: Union[str, None] = None,
            title: str = "",
            nb_samples: int = 100,
            shape: Optional[Tuple[int, int]] = None,
    ) -> None:
        """Plot samples of the current task, useful to check if everything is ok.

        :param path: If not None, save on disk at this path.
        :param title: The title of the figure.
        :param nb_samples: Amount of samples randomly selected.
        :param shape: Shape to resize the image before plotting.
        """
        raise NotImplementedError("we do not plot Tensor task set yet.")

    def __len__(self) -> int:
        """The amount of images in the current task."""
        return self._y.shape[0]

    def get_random_samples(self, nb_samples):
        nb_tot_samples = self._x.shape[0]
        indexes = np.random.randint(0, nb_tot_samples, nb_samples)
        return self.get_samples(indexes)

    def get_samples(self, indexes):
        samples, targets, tasks = [], [], []

        for index in indexes:
            # we need to use __getitem__ to have the transform used
            sample, y, t = self[index]
            samples.append(sample)
            targets.append(y)
            tasks.append(t)

        return _tensorize_list(samples), _tensorize_list(targets), _tensorize_list(tasks)

    def get_raw_samples(self, indexes=None):
        """Get samples without preprocessing, for split train/val for example."""
        if indexes is None:
            return self._x, self._y, self._t
        return self._x[indexes], self._y[indexes], self._t[indexes]

    def get_sample(self, index: int) -> np.ndarray:
        """Returns the tensor corresponding to the given `index`.

        :param index: Index to query the image.
        :return: A Pillow image.
        """
        x = self._x[index]
        if not torch.is_tensor(x):
            x = torch.tensor(x)
        return x

    def __getitem__(self, index: int) -> Tuple[np.ndarray, int, int]:
        """Method used by PyTorch's DataLoaders to query a sample and its target."""
        x = self.get_sample(index)
        y = self._y[index]
        t = self._t[index]

        if self.target_trsf is not None:
            y = self.get_task_target_trsf(t)(y)

        return x, y, t

    def get_task_trsf(self, t: int):
        if isinstance(self.trsf, list):
            return self.trsf[t]
        return self.trsf

    def get_task_target_trsf(self, t: int):
        if isinstance(self.target_trsf, list):
            return self.target_trsf[t]
        return self.target_trsf
