import enum
from copy import copy
from typing import Tuple, Union, Optional, List

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset as TorchDataset
from torchvision import transforms
from continuum.viz import plot_samples
from continuum.tasks.base import TaskType
from continuum.tasks.image_array_task_set import ArrayTaskSet


class PathTaskSet(ArrayTaskSet):
    """A task dataset returned by the CLLoader specialized into array of image's path to images.

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
        self.data_type = TaskType.IMAGE_PATH

    def get_sample(self, index: int) -> np.ndarray:
        """Returns a Pillow image corresponding to the given `index`.

        :param index: Index to query the image.
        :return: A Pillow image.
        """
        x = self._x[index]
        x = Image.open(x).convert("RGB")
        return x
