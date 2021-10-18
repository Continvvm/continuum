import os
from typing import Tuple

import numpy as np
from sklearn.model_selection import train_test_split
from torchvision import datasets as torchdata

from continuum.datasets import ImageFolderDataset
from continuum.download import download_file_from_google_drive, untar
from continuum.tasks import TaskType


class VLCS(ImageFolderDataset):
    """VLCS dataset group.

    Contain Caltech101, LabelMe, SUN09, and VOC2007. Each made of 5 classes:
    bird, car, chair, dog, and person.

    * Unbiased Metric Learning: On the Utilization of Multiple Datasets and Web Images for Softening Bias
      Fang, Xu, and Rockmore.
      ICCV 2013
    """
    images_gdrive_id = "1skwblH1_okBwxWxmRsp9_qi15hyPpxg8"

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
        if not os.path.exists(os.path.join(self.data_path, "VLCS")):
            tar_path = os.path.join(self.data_path, "VLCS.tar.gz")

            if not os.path.exists(tar_path):
                print("Downloading zip images archive...", end=' ')
                download_file_from_google_drive(self.images_gdrive_id, tar_path)
                print('Done!')

            print('Extracting archive...', end=' ')
            untar(tar_path)
            print('Done!')

    def get_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        domains = ["Caltech101", "LabelMe", "SUN09", "VOC2007"]

        full_x, full_y, full_t = [], [], []

        for domain_id, domain_name in enumerate(domains):
            dataset = torchdata.ImageFolder(os.path.join(self.data_path, "VLCS", domain_name))
            x, y, _ = self._format(dataset.imgs)
            x_train, x_test, y_train, y_test = train_test_split(
                x, y,
                test_size=self.test_split,
                random_state=self.random_seed
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
