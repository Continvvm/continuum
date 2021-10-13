from typing import Tuple, Union, Optional, List

import h5py
import numpy as np
from PIL import Image
import torch
from torchvision import transforms

from continuum.tasks.base import TaskType
from continuum.tasks.image_path_task_set import PathTaskSet


class H5TaskSet(PathTaskSet):
    """A task dataset returned by the CLLoader specialized into h5 data .

    :param x: The data, either image-arrays or paths to images saved on disk.
    :param y: The targets, not one-hot encoded.
    :param t: The task id of each sample.
    :param trsf: The transformations to apply on the images.
    :param target_trsf: The transformations to apply on the labels.
    :param data_index_index: data index of the current task (it makes possible to distinguish data of the current task
    from data from other tasks that are in the same h5 file.)
    """

    def __init__(
            self,
            x: str,
            y: np.ndarray,
            t: np.ndarray,
            trsf: Union[transforms.Compose, List[transforms.Compose]],
            target_trsf: Optional[Union[transforms.Compose, List[transforms.Compose]]] = None,
            bounding_boxes: Optional[np.ndarray] = None,
            data_indexes: np.ndarray = None
    ):

        self.h5_filename = x
        self._size_task_set = None
        self.data_type = TaskType.H5
        self.data_indexes = data_indexes

        if data_indexes is not None:
            self._size_task_set = len(data_indexes)
        else:
            with h5py.File(self.h5_filename, 'r') as hf:
                self._size_task_set = hf['y'].shape[0]

        super().__init__(self.h5_filename, y, t, trsf, target_trsf, bounding_boxes=bounding_boxes)

    def __len__(self) -> int:
        """The amount of images in the current task."""
        return self._size_task_set

    def __getitem__(self, index: int) -> Tuple[np.ndarray, int, int]:
        """Method used by PyTorch's DataLoaders to query a sample and its target."""
        x, y, t = None, None, None

        if self.data_indexes is not None:
            # the index is index data in the task not in the full dataset
            index = self.data_indexes[index]

        with h5py.File(self.h5_filename, 'r') as hf:
            x = hf['x'][index]
            y = hf['y'][index]
            if 't' in hf.keys():
                t = hf['t'][index]
            else:
                t = -1

        if self.bounding_boxes is not None:
            bbox = self.bounding_boxes[index]
            x = x.crop((
                max(bbox[0], 0),  # x1
                max(bbox[1], 0),  # y1
                min(bbox[2], x.size[0]),  # x2
                min(bbox[3], x.size[1]),  # y2
            ))

        if isinstance(x, str):
            # x = Image.open(x).convert("RGB")
            raise NotImplementedError("H5 taskset are not yet compatible to path array.")

        x, y, t = self._prepare_data(x, y, t)
        return x, y, t
