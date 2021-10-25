from typing import List, Tuple, Union

import numpy as np
from continuum.datasets import _ContinuumDataset
from continuum.tasks import TaskType
from torchvision import datasets as torchdata


class STL10(_ContinuumDataset):
    """STL10 Dataset.

    - 10 classes
    - 500 training images, 800 test images
    - 96x96x3

    * An Analysis of Single Layer Networks in Unsupervised Feature Learning
      Adam Coates, Honglak Lee, Andrew Y. Ng
      AISTATS 2011
    """
    def __init__(
            self, data_path: str = "", train: Union[bool, str] = True, download: bool = True):

        super().__init__(data_path=data_path, train=train, download=download)

        if isinstance(train, bool):
            train = "train" if train else "test"
        self.train = train

        self.dataset = torchdata.STL10(
            self.data_path,
            download=self.download,
            split=self.train)

    @property
    def data_type(self) -> TaskType:
        return TaskType.IMAGE_ARRAY

    def get_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        x, y = np.array(self.dataset.data), np.array(self.dataset.labels)
        return x.transpose(0, 2, 3, 1), y, None
