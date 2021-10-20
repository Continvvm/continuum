import enum
from copy import copy
from typing import Tuple, Union, Optional, List

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from continuum.viz import plot_samples
from continuum.tasks.base import BaseTaskSet, _tensorize_list, TaskType


class ArrayTaskSet(BaseTaskSet):
    """A task dataset returned by the CLLoader specialized into numpy/torch image arrays data.

    :param x: The data, either image-arrays or paths to images saved on disk.
    :param y: The targets, not one-hot encoded.
    :param t: The task id of each sample.
    :param trsf: The transformations to apply on the images.
    :param target_trsf: The transformations to apply on the labels.
    :param bounding_boxes: The bounding boxes annotations to crop images
    """

    def __init__(
            self,
            x: np.ndarray,
            y: np.ndarray,
            t: np.ndarray,
            trsf: Union[transforms.Compose, List[transforms.Compose]],
            target_trsf: Optional[Union[transforms.Compose, List[transforms.Compose]]],
            bounding_boxes: Optional[np.ndarray] = None
    ):
        super().__init__(x, y, t, trsf, target_trsf, bounding_boxes=bounding_boxes)
        self.data_type = TaskType.IMAGE_ARRAY

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

    def get_samples(self, indexes):
        samples, targets, tasks = [], [], []

        w, h = None, None
        for index in indexes:
            # we need to use __getitem__ to have the transform used
            sample, y, t = self[index]

            # we check dimension of images
            if w is None:
                w, h = sample.shape[:2]
            elif w != sample.shape[0] or h != sample.shape[1]:
                raise Exception(
                    "Images dimension are inconsistent, resize them to a "
                    "common size using a transformation.\n"
                    "For example, give to the scenario you're using as `transformations` argument "
                    "the following: [transforms.Resize((224, 224)), transforms.ToTensor()]"
                )

            samples.append(sample)
            targets.append(y)
            tasks.append(t)

        return _tensorize_list(samples), _tensorize_list(targets), _tensorize_list(tasks)

    def get_sample(self, index: int) -> np.ndarray:
        """Returns a Pillow image corresponding to the given `index`.

        :param index: Index to query the image.
        :return: A Pillow image.
        """
        x = self._x[index]
        x = Image.fromarray(x.astype("uint8"))
        return x

    def __getitem__(self, index: int) -> Tuple[np.ndarray, int, int]:
        """Method used by PyTorch's DataLoaders to query a sample and its target."""
        x = self.get_sample(index)
        y = self._y[index]
        t = self._t[index]

        if self.bounding_boxes is not None:
            bbox = self.bounding_boxes[index]
            x = x.crop((
                max(bbox[0], 0),  # x1
                max(bbox[1], 0),  # y1
                min(bbox[2], x.size[0]),  # x2
                min(bbox[3], x.size[1]),  # y2
            ))

        x, y, t = self._prepare_data(x, y, t)

        if self.target_trsf is not None:
            y = self.get_task_target_trsf(t)(y)

        return x, y, t

    def _prepare_data(self, x, y, t):
        if self.trsf is not None:
            x = self.get_task_trsf(t)(x)
        if not isinstance(x, torch.Tensor):
            x = self._to_tensor(x)
        return x, y, t
