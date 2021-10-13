from typing import Tuple, Union, Optional, List

import h5py
import numpy as np
from PIL import Image
import torch
from torchvision import transforms

from continuum.tasks import ArrayTaskSet, TaskType


class H5TaskSet(ArrayTaskSet):
    """A task dataset returned by the CLLoader.

    :param dataset_filename: a path to the dataset
    :param trsf: The transformations to apply on the images.
    :param data_type: Type of the data, either "image_path", "image_array",
                      "text", "tensor" or "segmentation".
    """

    def __init__(
            self,
            dataset: TaskType.H5,
            trsf: Union[transforms.Compose, List[transforms.Compose]],
            target_trsf: Optional[Union[transforms.Compose, List[transforms.Compose]]] = None,
            bounding_boxes: Optional[np.ndarray] = None):

        dataset_filename, _, _ = dataset.get_data()
        self.h5_filename = dataset_filename
        self._size_dataset = None
        with h5py.File(self.h5_filename, 'r') as hf:
            self._size_dataset = hf['y'].shape[0]

        super().__init__(x=dataset_filename,
                            y=None,
                            t=None,
                            trsf=trsf,
                            target_trsf = target_trsf)

    def __len__(self) -> int:
        """The amount of images in the current task."""
        return self._size_dataset

    def __getitem__(self, index: int) -> Tuple[np.ndarray, int, int]:
        """Method used by PyTorch's DataLoaders to query a sample and its target."""
        x, y, t = None, None, None
        with h5py.File(self.h5_filename, 'r') as hf:
            x = hf['x'][index]
            y = hf['y'][index]
            t = hf['t'][index]

        if isinstance(x, str):
            x = Image.open(x).convert("RGB")
        elif isinstance(x, torch.Tensor):
            #not thing to do here
            pass
        elif isinstance(x, np.ndarray):
            x = self._to_tensor(x)
        else:
            raise NotImplementedError(f"The type {type(x)} is not compatible with h5 task set.")

        x, y, t = self._prepare_data(x, y, t)
        return x, y, t
