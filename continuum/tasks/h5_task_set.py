from typing import Tuple, Union, Optional, List

import h5py
import numpy as np
from PIL import Image
import torch
from torchvision import transforms

from continuum.tasks.base import TaskType
from continuum.tasks.image_path_task_set import PathTaskSet


class H5TaskSet(PathTaskSet):
    """A task dataset returned by the CLLoader.

    :param dataset_filename: a path to the dataset
    :param trsf: The transformations to apply on the images.
    :param data_type: Type of the data, either "image_path", "image_array",
                      "text", "tensor" or "segmentation".
    """

    def __init__(
            self,
            x: str,
            y: np.ndarray,
            t: np.ndarray,
            trsf: Union[transforms.Compose, List[transforms.Compose]],
            target_trsf: Optional[Union[transforms.Compose, List[transforms.Compose]]] = None,
            bounding_boxes: Optional[np.ndarray] = None):

        self.h5_filename = x
        self._size_dataset = None
        self.data_type = TaskType.H5
        with h5py.File(self.h5_filename, 'r') as hf:
            self._size_dataset = hf['y'].shape[0]

        super().__init__(x=self.h5_filename,
                         y=None,
                         t=None,
                         trsf=trsf,
                         target_trsf=target_trsf)

    def __len__(self) -> int:
        """The amount of images in the current task."""
        return self._size_dataset

    def __getitem__(self, index: int) -> Tuple[np.ndarray, int, int]:
        """Method used by PyTorch's DataLoaders to query a sample and its target."""
        x, y, t = None, None, None
        with h5py.File(self.h5_filename, 'r') as hf:
            x = hf['x'][index]
            y = hf['y'][index]
            if 't' in hf.keys():
                t = hf['t'][index]
            else:
                t = -1

        if isinstance(x, str):
            # x = Image.open(x).convert("RGB")
            raise NotImplementedError("H5 taskset are not yet compatible to path array.")

        x, y, t = self._prepare_data(x, y, t)
        return x, y, t
