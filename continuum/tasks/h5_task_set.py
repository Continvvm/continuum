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
        self._y = y
        self._t = t

        if data_indexes is not None:
            self._size_task_set = len(data_indexes)
            assert len(data_indexes) == len(y)

        super().__init__(self.h5_filename, y, t, trsf, target_trsf, bounding_boxes=bounding_boxes)

    def get_sample(self, index):
        with h5py.File(self.h5_filename, 'r') as hf:
            x = hf['x'][index]
        return x

    def __getitem__(self, index):
        remapped_index = self.data_indexes[index]  # Note that 'data_indexes' should simply be called 'indexes'
        return super().__getitem__(remapped_index)
