from typing import List, Tuple, Union

import numpy as np
from torchvision import datasets as torchdata

from continuum.datasets import _ContinuumDataset


class STL10(_ContinuumDataset):
    def __init__(
            self, data_path: str = "", train: Union[bool, str] = True, download: bool = True):

        super().__init__(data_path=data_path, train=train, download=download)

        if isinstance(train, bool):
            train = "train" if train else "test"
        self.train = train

        self.dataset = torchdata.STL10(
            self.data_path,
            download=self.download,
            split=self.train
    )

    def get_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        x, y = np.array(self.dataset.data), np.array(self.dataset.labels)
        return x.transpose(0, 2, 3, 1), y, None
