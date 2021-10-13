import enum
from copy import copy
from typing import Tuple, Union, Optional, List

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset as TorchDataset
from torchvision import transforms

from continuum.viz import plot_samples
from continuum.tasks.base import BaseTaskSet, TaskType
from continuum.tasks.image_array_task_set import ArrayTaskSet
from continuum.tasks.image_path_task_set import PathTaskSet
from continuum.tasks.segmentation_task_set import SegmentationTaskSet
from continuum.tasks.text_task_set import TextTaskSet
from continuum.tasks.h5_task_set import H5TaskSet


def TaskSet(x: np.ndarray,
            y: np.ndarray,
            t: np.ndarray,
            trsf: Union[transforms.Compose, List[transforms.Compose]],
            target_trsf: Optional[Union[transforms.Compose, List[transforms.Compose]]] = None,
            data_type: TaskType = TaskType.IMAGE_ARRAY,
            bounding_boxes: Optional[np.ndarray] = None):

    if data_type == TaskType.TEXT:
        assert bounding_boxes is None, print("bounding_boxes are not compatible with TaskType.TEXT")
        task_set = TextTaskSet(x=x, y=y, t=t, trsf=trsf, target_trsf=target_trsf)
    elif data_type == TaskType.IMAGE_ARRAY:
        assert bounding_boxes is None, print("bounding_boxes are not compatible with TaskType.IMAGE_ARRAY")
        task_set = ArrayTaskSet(x=x, y=y, t=t, trsf=trsf, target_trsf=target_trsf)
    elif data_type == TaskType.IMAGE_PATH:
        assert bounding_boxes is None, print("bounding_boxes are not compatible with TaskType.IMAGE_PATH")
        task_set = PathTaskSet(x=x, y=y, t=t, trsf=trsf, target_trsf=target_trsf)
    elif data_type == TaskType.SEGMENTATION:
        task_set = SegmentationTaskSet(x=x, y=y, t=t, trsf=trsf, target_trsf=target_trsf, bounding_boxes=bounding_boxes)
    elif data_type == TaskType.TENSOR:
        assert bounding_boxes is None, print("bounding_boxes are not compatible with TaskType.TENSOR")
        task_set = BaseTaskSet(x=x, y=y, t=t, trsf=trsf, target_trsf=target_trsf)
    elif data_type == TaskType.H5:
        if bounding_boxes is not None:
            raise NotImplementedError("h5 datasets are not yet compatible with bounding_boxes")
        task_set = H5TaskSet(x=x, y=y, t=t, trsf=trsf, target_trsf=target_trsf)
    else:
        raise AssertionError(f"No TaskSet for data_type {data_type}")

    return task_set

def _tensorize_list(x):
    if isinstance(x[0], torch.Tensor):
        return torch.stack(x)
    return torch.tensor(x)
