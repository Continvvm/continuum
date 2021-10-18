import os
from typing import Tuple

import numpy as np
from sklearn.model_selection import train_test_split
from torchvision import datasets as torchdata

from continuum.datasets import ImageFolderDataset
from continuum.download import download, unzip
from continuum.tasks import TaskType


class AwA2(ImageFolderDataset):
    """AwA2 dataset.

    * Zero-Shot Learning - A Comprehensive Evaluation of the Good, the Bad and the Ugly
      Y. Xian, C. H. Lampert, B. Schiele, Z. Akata
      TPAMI 2018
    """
    images_url = "https://cvml.ist.ac.at/AwA2/AwA2-data.zip"
    split_v2_url = "http://datasets.d2.mpi-inf.mpg.de/xian/xlsa17.zip"

    def __init__(self, data_path, train: bool = True, download: bool = True, test_split: float = 0.2,
                 random_seed=1):
        self._attributes = None
        self.test_split = test_split
        self.random_seed = random_seed
        super().__init__(data_path, train, download)

    @property
    def attributes(self):
        if self._attributes is None:
            att = np.loadtxt(
                os.path.join(
                    self.data_path, "Animals_with_Attributes2",
                    "predicate-matrix-continuous.txt"
                )
            )
            self._attributes = att / np.linalg.norm(att, axis=-1, keepdims=True)

        return self._attributes

    @property
    def data_type(self) -> TaskType:
        return TaskType.IMAGE_PATH

    def _download(self):
        if not os.path.exists(os.path.join(self.data_path, "Animals_with_Attributes2")):
            zip_path = os.path.join(self.data_path, "AwA2-data.zip")

            if not os.path.exists(zip_path):
                print("Downloading zip images archive...", end=' ')
                download(self.images_url, self.data_path)
                print('Done!')

            print('Extracting archive...', end=' ')
            unzip(zip_path)
            print('Done!')

        if not os.path.exists(os.path.join(self.data_path, "xlsa17")):
            zip_path = os.path.join(self.data_path, "xlsa17.zip")

            if not os.path.exists(zip_path):
                print("Downloading zip split archive...", end=' ')
                download(self.split_v2_url, self.data_path)
                print('Done!')

            print('Extracting archive...', end=' ')
            unzip(zip_path)
            print('Done!')

    def get_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        dataset = torchdata.ImageFolder(os.path.join(self.data_path, "Animals_with_Attributes2", "JPEGImages"))
        x, y, _ = self._format(dataset.imgs)

        x_train, x_test, y_train, y_test = train_test_split(
            x, y,
            test_size=self.test_split,
            random_state=self.random_seed
        )

        if self.train:
            return x_train, y_train, None
        return x_test, y_test, None
