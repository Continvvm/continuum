import os
from typing import Tuple

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from torchvision import datasets as torchdata

from continuum.datasets import ImageFolderDataset
from continuum.download import download, untar, unzip


class TerraIncognita(ImageFolderDataset):
    """TerraIncognita dataset group.

    Filtered according to DomainBed rule, whose code was largely used here:
    https://github.com/facebookresearch/DomainBed/blob/main/domainbed/scripts/download.py#L167

    Contain 4 different domains (art painting, cartoon, photo, and sketch).
    Each made of 7 classes (dog, elephant, giraffe, guitar, horse, house, and person).

    * Recognition in Terra Incognita
      Beery et al.
      ECCV 2018
    """
    images_url = "https://lilablobssc.blob.core.windows.net/caltechcameratraps/eccv_18_all_images_sm.tar.gz"
    json_url = "https://lilablobssc.blob.core.windows.net/caltechcameratraps/labels/caltech_camera_traps.json.zip"

    def __init__(self, data_path, train: bool = True, download: bool = True,
                 test_split: float = 0.2, random_seed: int = 1):
        self._attributes = None
        self.test_split = test_split
        self.random_seed = random_seed
        super().__init__(data_path, train, download)

    @property
    def data_type(self):
        return "image_data_path"

    def _download(self):
        if not os.path.exists(os.path.join(self.data_path, "eccv_18_all_images_sm")):
            tar_path = os.path.join(self.data_path, "eccv_18_all_images_sm.tar.gz")
            if not os.path.exists(tar_path):
                print("Downloading images archive...", end=" ")
                download(self.images_url, self.data_path)
                print("Done!")
            print('Extracting archive...', end=' ')
            untar(zip_path)
            print('Done!')

        if not os.path.exists(os.path.join(self.data_path, "caltech_camera_traps.json")):
            zip_path = os.path.join(self.data_path, "caltech_camera_traps.json.zip")
            if not os.path.exists(zip_path):
                print("Downloading json archive...", end=" ")
                download(self.json_url, self.data_path)
                print("Done!")
            print('Extracting archive...', end=' ')
            unzip(zip_path)
            print('Done!')

    def get_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """See https://github.com/facebookresearch/DomainBed/blob/main/domainbed/scripts/download.py#L167"""
        domains = ["art_painting", "cartoon", "photo", "sketch"]

        full_x, full_y, full_t = [], [], []

        for domain_id, domain_name in enumerate(domains):
            dataset = torchdata.ImageFolder(os.path.join(self.data_path, "kfold", domain_name))
            x, y, _ = self._format(dataset.imgs)
            x_train, x_test, y_test, y_train = train_test_split(
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
