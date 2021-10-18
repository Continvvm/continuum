import os
from typing import Tuple

import numpy as np
from sklearn.model_selection import train_test_split
from torchvision import datasets as torchdata

from continuum.datasets import ImageFolderDataset
from continuum.download import download, unzip
from continuum.tasks import TaskType


class EuroSAT(ImageFolderDataset):
    """EuroSAT dataset --RGB version-- !.

    Satellite images with 10 classes and 27,000 labeled images.

    * Eurosat: A novel dataset and deep learning benchmark for land use and land cover classification
      Helber, Patrick and Bischke, Benjamin and Dengel, Andreas and Borth, Damian
      IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing 2019
    """
    images_url = "http://madm.dfki.de/files/sentinel/EuroSAT.zip"

    def __init__(self, data_path, train: bool = True, download: bool = True, test_split: float = 0.2,
                 random_seed=1):
        self.test_split = test_split
        self.random_seed = random_seed
        super().__init__(data_path, train, download)

    @property
    def data_type(self) -> TaskType:
        return TaskType.IMAGE_PATH

    def _download(self):
        if not os.path.exists(os.path.join(self.data_path, "2750")):
            zip_path = os.path.join(self.data_path, "EuroSAT.zip")

            if not os.path.exists(zip_path):
                print("Downloading zip images archive...", end=' ')
                download(self.images_url, self.data_path)
                print('Done!')

            print('Extracting archive...', end=' ')
            unzip(zip_path)
            print('Done!')

    def get_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        dataset = torchdata.ImageFolder(os.path.join(self.data_path, "2750"))
        x, y, _ = self._format(dataset.imgs)

        x_train, x_test, y_train, y_test = train_test_split(
            x, y,
            test_size=self.test_split,
            random_state=self.random_seed
        )

        if self.train:
            return x_train, y_train, None
        return x_test, y_test, None
