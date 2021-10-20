from typing import Tuple

import numpy as np
from torchvision import datasets as torchdata

from continuum.datasets import _ContinuumDataset


class SVHN(_ContinuumDataset):
    def __init__(
            self,
            data_path: str = "",
            train: bool = True,
            download: bool = True
        ):
        super().__init__(data_path=data_path, train=train, download=download)
        self.dataset = torchdata.SVHN(
            self.data_path,
            download=self.download,
            split="train" if train else "test"
        )

    def get_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        x, y = np.array(self.dataset.data), np.array(self.dataset.labels)
        x = x.transpose(0, 2, 3, 1)
        return x, y, None
