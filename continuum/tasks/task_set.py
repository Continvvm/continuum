from typing import List, Optional, Union

import numpy as np
from torchvision import transforms

from continuum.tasks.base import BaseTaskSet, TaskType
from continuum.tasks.image_array_task_set import ArrayTaskSet
from continuum.tasks.image_path_task_set import PathTaskSet
from continuum.tasks.segmentation_task_set import SegmentationTaskSet
from continuum.tasks.text_task_set import TextTaskSet
from continuum.tasks.h5_task_set import H5TaskSet


def TaskSet(x: np.ndarray,
            y: np.ndarray,
            t: np.ndarray,
            trsf: Union[transforms.Compose, List[transforms.Compose]] = None,
            target_trsf: Optional[Union[transforms.Compose, List[transforms.Compose]]] = None,
            data_type: TaskType = TaskType.IMAGE_ARRAY,
            bounding_boxes: Optional[np.ndarray] = None,
            data_indexes=None):
    if data_type == TaskType.TEXT:
        assert bounding_boxes is None, print("bounding_boxes are not compatible with TaskType.TEXT")
        task_set = TextTaskSet(x=x, y=y, t=t, trsf=trsf, target_trsf=target_trsf)
    elif data_type == TaskType.IMAGE_ARRAY:
        task_set = ArrayTaskSet(x=x, y=y, t=t, trsf=trsf, target_trsf=target_trsf, bounding_boxes=bounding_boxes)
    elif data_type == TaskType.IMAGE_PATH:
        task_set = PathTaskSet(x=x, y=y, t=t, trsf=trsf, target_trsf=target_trsf, bounding_boxes=bounding_boxes)
    elif data_type == TaskType.SEGMENTATION:
        task_set = SegmentationTaskSet(x=x, y=y, t=t, trsf=trsf, target_trsf=target_trsf, bounding_boxes=bounding_boxes)
    elif data_type == TaskType.TENSOR:
        assert bounding_boxes is None, print("bounding_boxes are not compatible with TaskType.TENSOR")
        task_set = BaseTaskSet(x=x, y=y, t=t, trsf=trsf, target_trsf=target_trsf)
    elif data_type == TaskType.H5:
        task_set = H5TaskSet(x=x, y=y, t=t, trsf=trsf,
                             target_trsf=target_trsf,
                             bounding_boxes=bounding_boxes,
                             data_indexes=data_indexes)
    else:
        raise AssertionError(f"No TaskSet for data_type {data_type}")

    return task_set
