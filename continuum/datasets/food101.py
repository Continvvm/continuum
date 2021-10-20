import os
from typing import Tuple, List

import numpy as np
from torchvision import datasets as torchdata

from continuum.datasets import ImageFolderDataset
from continuum.download import download, untar
from continuum.tasks import TaskType


class Food101(ImageFolderDataset):
    """Food101 dataset.

    * Food-101 – Mining Discriminative Components with Random Forests
      Lukas Bossard, Matthieu Guillaumin and Luc Van Gool
    """
    images_url = "http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz"

    def __init__(self, data_path, train: bool = True, download: bool = True):
        self._attributes = None
        super().__init__(data_path, train, download)

    @property
    def data_type(self) -> TaskType:
        return TaskType.IMAGE_PATH

    def _download(self):
        if not os.path.exists(os.path.join(self.data_path, "food-101")):
            archive_path = os.path.join(self.data_path, "food-101.tar.gz")

            if not os.path.exists(archive_path):
                print("Downloading images archive...", end=' ')
                download(self.images_url, self.data_path)
                print('Done!')

            print('Extracting archive...', end=' ')
            untar(archive_path)
            print('Done!')

    def get_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        dataset = torchdata.ImageFolder(os.path.join(self.data_path, "food-101", "images"))
        x, y, _ = self._format(dataset.imgs)

        test_ids = set()
        with open(os.path.join(self.data_path, "food-101", "meta", "test.txt")) as f:
            for line in f:
                test_ids.add(line.split("/")[-1].strip())

        final_x, final_y = [], []
        for path, label in zip(x, y):
            image_id = str(path).split("/")[-1][:-5]

            if (self.train and image_id not in test_ids) or \
                (not self.train and image_id in test_ids):
                final_x.append(path)
                final_y.append(label)

        x, y = np.array(final_x), np.array(final_y)
        return x, y, None