import os
from typing import Tuple

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from torchvision import datasets as torchdata

from continuum.datasets import ImageFolderDataset
from continuum.download import unzip
from continuum.tasks import TaskType


class OfficeHome(ImageFolderDataset):
    """OfficeHome dataset group.

    Contain 4 different domains (art, clipart, product, and real world).
    Each made of 65 classes.

    * Deep Hashing Network for Unsupervised Domain Adaptation
      Venkateswara et al.
      CVPR 2017
    """
    def __init__(self, data_path, train: bool = True, download: bool = True,
                 test_split: float = 0.2, random_seed: int = 1):
        self._attributes = None
        self.test_split = test_split
        self.random_seed = random_seed
        super().__init__(data_path, train, download)

    @property
    def data_type(self) -> TaskType:
        return TaskType.IMAGE_PATH

    def _download(self):
        if not os.path.exists(os.path.join(self.data_path, "OfficeHomeDataset_10072016")):
            zip_path = os.path.join(self.data_path, "OfficeHomeDataset_10072016.zip")

            if not os.path.exists(zip_path):
                print(
                    "You need to download yourself this dataset, at the following url"
                    " https://drive.google.com/uc?id=0B81rNlvomiwed0V1YUxQdC1uOTg"
                )

            print('Extracting archive...', end=' ')
            unzip(zip_path)
            print('Done!')

    def get_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        domains = ["Art", "Clipart", "Product", "Real World"]

        full_x, full_y, full_t = [], [], []

        for domain_id, domain_name in enumerate(domains):
            dataset = torchdata.ImageFolder(os.path.join(self.data_path, "OfficeHomeDataset_10072016", domain_name))
            x, y, _ = self._format(dataset.imgs)
            x_train, x_test, y_train, y_test = train_test_split(
                x, y,
                test_size=self.test_split,
                random_state=1
            )

            if self.train:
                full_x.append(x_train),
                full_y.append(y_train)
            else:
                full_x.append(x_test)
                full_y.append(y_test)
            full_t.append(np.ones_like(full_y[-1]) * domain_id)

        x = np.concatenate(full_x)
        y = np.concatenate(full_y)
        t = np.concatenate(full_t)
        return x, y, t
