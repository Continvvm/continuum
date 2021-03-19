from copy import copy
from typing import Tuple, Union, Optional

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
    :param data_type: Type of the data, either "image_path", "image_array",
                      "text", or "segmentation".
    """

    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        t: np.ndarray,
        trsf: transforms.Compose,
        target_trsf: Optional[transforms.Compose] = None,
        data_type: str = "image_array",
        bounding_boxes: Optional[np.ndarray] = None
    ):
        self._x, self._y, self._t = x, y, t

        # if task index are not provided t is always -1
        if self._t is None:
            self._t = -1 * np.ones_like(y)

        self.trsf = trsf
        self.target_trsf = target_trsf
        self.data_type = data_type
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
                    f"Invalid data type {task_set.data_type} != {self.data_type}"
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
        plot_samples(self, title=title, path=path, nb_samples=nb_samples,
                     shape=shape, data_type=self.data_type)

    def __len__(self) -> int:
        """The amount of images in the current task."""
        return self._x.shape[0]

    def get_random_samples(self, nb_samples):
        nb_tot_samples = self._x.shape[0]
        indexes = np.random.randint(0, nb_tot_samples, nb_samples)
        return self.get_samples(indexes)

    def get_samples(self, indexes):
        images, targets, tasks = [], [], []

        w, h = None, None
        for index in indexes:
            # we need to use __getitem__ to have the transform used
            img, y, t = self[index]

            if w is None:
                w, h = img.shape[:2]
            elif w != img.shape[0] or h != img.shape[1]:
                raise Exception(
                    "Images dimension are inconsistent, resize them to a "
                    "common size using a transformation.\n"
                    "For example, give to the scenario you're using as `transformations` argument "
                    "the following: [transforms.Resize((224, 224)), transforms.ToTensor()]"
                )

            images.append(img)
            targets.append(y)
            tasks.append(t)

        return _tensorize_list(images), _tensorize_list(targets), _tensorize_list(tasks)

    def get_raw_samples(self, indexes):
        """Get samples without preprocessing, for split train/val for example."""
        return self._x[indexes], self._y[indexes], self._t[indexes]

    def get_sample(self, index: int) -> np.ndarray:
        """Returns a Pillow image corresponding to the given `index`.

        :param index: Index to query the image.
        :return: A Pillow image.
        """
        x = self._x[index]

        if self.data_type in ("image_path", "segmentation"):
            x = Image.open(x).convert("RGB")
        elif self.data_type == "image_array":
            x = Image.fromarray(x.astype("uint8"))
        elif self.data_type == "text":
            pass

        return x

    def __getitem__(self, index: int) -> Tuple[np.ndarray, int, int]:
        """Method used by PyTorch's DataLoaders to query a sample and its target."""
        x = self.get_sample(index)
        y = self._y[index]
        t = self._t[index]

        if self.bounding_boxes is not None:
            bbox = self.bounding_boxes[index]
            x = x.crop((
                max(bbox[0], 0),               # x1
                max(bbox[1], 0),               # y1
                min(bbox[2], x.size[0]),  # x2
                min(bbox[3], x.size[1]),  # y2
            ))

        if self.data_type == "text":
            x, y, t = self._prepare_text(x, y, t)
        elif self.data_type == "segmentation":
            x, y, t = self._prepare_segmentation(x, y, t)
        else:
            x, y, t = self._prepare(x, y, t)

        if self.target_trsf is not None:
            y = self.target_trsf(y)

        return x, y, t

    def _prepare(self, x, y, t):
        if self.trsf is not None:
            x = self.trsf(x)
        if not isinstance(x, torch.Tensor):
            x = self._to_tensor(x)

        return x, y, t

    def _prepare_segmentation(self, x, y, t):
        y = Image.open(y)
        if self.trsf is not None:
            x, y = self.trsf(x, y)

        if not isinstance(x, torch.Tensor):
            x = self._to_tensor(x)
        if not isinstance(y, torch.Tensor):
            y = self._to_tensor(y)

        return x, y, t

    def _prepare_text(self, x, y, t):
        # Nothing in particular for now, TODO latter
        return x, y, t


def _tensorize_list(x):
    if isinstance(x[0], torch.Tensor):
        return torch.stack(x)
    return torch.tensor(x)
