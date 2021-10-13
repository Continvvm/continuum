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
from continuum.tasks.image_path_task_set import PathTaskSet

class SegmentationTaskSet(PathTaskSet):
    """A task dataset returned by the CLLoader specialized into segmentation data.

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
            target_trsf: Optional[Union[transforms.Compose, List[transforms.Compose]]] = None,
            bounding_boxes: Optional[np.ndarray] = None
    ):
        super().__init__(x, y, t, trsf, target_trsf)
        self.data_type = TaskType.SEGMENTATION
        self.bounding_boxes = bounding_boxes


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
        y = Image.open(y)
        if self.trsf is not None:
            x, y = self.get_task_trsf(t)(x, y)

        if not isinstance(x, torch.Tensor):
            x = self._to_tensor(x)
        if not isinstance(y, torch.Tensor):
            y = self._to_tensor(y)

        return x, y, t