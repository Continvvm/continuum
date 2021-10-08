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
            assert bounding_boxes is None, print("bounding_boxes are not compatible with TaskType.TEXT")
            task_set = ArrayTaskSet(x=x, y=y, t=t, trsf=trsf, target_trsf=target_trsf)
    elif data_type == TaskType.IMAGE_PATH:
            assert bounding_boxes is None, print("bounding_boxes are not compatible with TaskType.TEXT")
            task_set = PathTaskSet(x=x, y=y, t=t, trsf=trsf, target_trsf=target_trsf)
    elif data_type == TaskType.SEGMENTATION:
            task_set = SegmentationTaskSet(x=x, y=y, t=t, trsf=trsf, target_trsf=target_trsf, bounding_boxes=bounding_boxes)
    else:
        task_set = BaseTaskSet(x=x, y=y, t=t, trsf=trsf, target_trsf=target_trsf, data_type=data_type,
                     bounding_boxes=bounding_boxes)
    return task_set

def _tensorize_list(x):
    if isinstance(x[0], torch.Tensor):
        return torch.stack(x)
    return torch.tensor(x)
