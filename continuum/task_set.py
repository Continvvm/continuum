from typing import Tuple, Union

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset as TorchDataset
from torchvision import transforms

from continuum.viz import plot_samples


class TaskSet(TorchDataset):
    """A task dataset returned by the CLLoader.

    :param x: The data, either image-arrays or paths to images saved on disk.
    :param y: The targets, not one-hot encoded.
    :param t: The task id of each sample.
    :param trsf: The transformations to apply on the images.
    :param data_type: Type of the data, either "image_path", "image_array", or "text".
    """

    def __init__(
            self,
            x: np.ndarray,
            y: np.ndarray,
            t: np.ndarray,
            trsf: transforms.Compose,
            data_type: str = "image_array"
    ):

        self._x, self._y, self._t = x, y, t
        self.trsf = trsf
        self.data_type = data_type

    @property
    def nb_classes(self):
        """The number of classes contained in the current task."""
        return len(np.unique(self._y))

    def get_classes(self):
        """Array of all classes contained in the current task."""
        return np.unique(self._y)

    def add_samples(
            self, x: np.ndarray, y: np.ndarray, t: Union[None, np.ndarray] = None
    ):
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
            shape=None
    ) -> None:
        """Plot samples of the current task, useful to check if everything is ok.

        :param path: If not None, save on disk at this path.
        :param title: The title of the figure.
        :param nb_samples: Amount of samples randomly selected.
        :param shape: Shape to resize the image before plotting.
        """
        plot_samples(self, title=title, path=path, nb_samples=nb_samples, shape=shape)

    def __len__(self) -> int:
        """The amount of images in the current task."""
        return self._x.shape[0]

    def get_random_samples(self, nb_samples):
        nb_tot_samples = self._x.shape[0]
        indexes = np.random.randint(0, nb_tot_samples, nb_samples)
        return self.get_samples(indexes)

    def get_samples(self, indexes):
        images, targets, tasks = [], [], []

        for index in indexes:
            # we need to use __getitem__ to have the transform used
            img, y, t = self[index]

            images.append(img)
            targets.append(y)
            tasks.append(t)

        return torch.stack(images), torch.Tensor(targets), torch.Tensor(tasks)

    def get_sample(self, index: int) -> np.ndarray:
        """Returns a Pillow image corresponding to the given `index`.

        :param index: Index to query the image.
        :return: A Pillow image.
        """
        x = self._x[index]

        if self.data_type == "image_path":
            x = Image.open(x).convert("RGB")
        elif self.data_type == "image_array":
            x = Image.fromarray(x.astype("uint8"))
        elif self.data_type == "text":
            pass

        return x

    def __getitem__(self, index: int) -> Tuple[np.ndarray, int, int]:
        """Method used by PyTorch's DataLoaders to query a sample and its target."""
        img = self.get_sample(index)
        y = self._y[index]
        t = self._t[index]

        if self.trsf is not None:
            img = self.trsf(img)

        # we impose output data to be Tensor
        if not isinstance(img, torch.Tensor):
            img = transforms.ToTensor()(img)

        return img, y, t

    def get_raw_samples(self, indexes):
        """Get samples without preprocessing, for split train/val for example."""
        return self._x[indexes], self._y[indexes], self._t[indexes]


def split_train_val(dataset: TaskSet, val_split: float = 0.1) -> Tuple[TaskSet, TaskSet]:
    """Split train dataset into two datasets, one for training and one for validation.

    :param dataset: A torch dataset, with .x and .y attributes.
    :param val_split: Percentage to allocate for validation, between [0, 1[.
    :return: A tuple a dataset, respectively for train and validation.
    """
    random_state = np.random.RandomState(seed=1)
    indexes = np.arange(len(dataset))
    random_state.shuffle(indexes)

    train_indexes = indexes[int(val_split * len(indexes)):]
    val_indexes = indexes[:int(val_split * len(indexes))]

    x_train, y_train, t_train = dataset.get_raw_samples(train_indexes)
    train_dataset = TaskSet(x_train, y_train, t_train, dataset.trsf, dataset.data_type)

    x_val, y_val, t_val = dataset.get_raw_samples(val_indexes)
    val_dataset = TaskSet(x_val, y_val, t_val, dataset.trsf, dataset.data_type)

    return train_dataset, val_dataset
