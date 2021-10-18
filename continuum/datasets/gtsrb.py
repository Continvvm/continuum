import os
from typing import Tuple

import numpy as np
from torchvision import datasets as torchdata

from continuum.datasets import ImageFolderDataset
from continuum.download import download, unzip
from continuum.tasks import TaskType


class GTSRB(ImageFolderDataset):
    """German Traffic Sign Recognition Benchmark (GTSRB) dataset.

    43 classes of relatively small images (usually less than 100 pixels wide).

    * Detection of Traffic Signs in Real-World Images: The {G}erman {T}raffic {S}ign {D}etection {B}enchmark}
      Sebastian Houben and Johannes Stallkamp and Jan Salmen and Marc Schlipsing and Christian Igel
      IJCNN 2013
    """
    train_images_url = "https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Training_Images.zip"
    test_images_url = "https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Test_Images.zip"
    test_gt_url = "https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Test_GT.zip"

    def __init__(self, data_path, train: bool = True, download: bool = True):
        super().__init__(data_path, train, download)

    @property
    def data_type(self) -> TaskType:
        return TaskType.IMAGE_PATH

    def _download(self):
        paths = [
            os.path.join(self.data_path, "GTSRB", "Final_Training"),
            os.path.join(self.data_path, "GTSRB", "Final_Test"),
            os.path.join(self.data_path, "GT-final_test.csv")
        ]
        zips = [
            os.path.join(self.data_path, "GTSRB_Final_Training_Images.zip"),
            os.path.join(self.data_path, "GTSRB_Final_Test_Images.zip"),
            os.path.join(self.data_path, "GTSRB_Final_Test_GT.zip")
        ]
        urls = [
            self.train_images_url, self.test_images_url, self.test_gt_url
        ]

        for p, z, u in zip(paths, zips, urls):
            if not os.path.exists(p):
                if not os.path.exists(z):
                    print("Downloading images archive...", end=' ')
                    download(u, self.data_path)
                    print('Done!')

                print('Extracting archive...', end=' ')
                unzip(z)
                print('Done!')

    def get_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.train:
            dataset = torchdata.ImageFolder(os.path.join(self.data_path, "GTSRB", "Final_Training", "Images"))
            x, y, _ = self._format(dataset.imgs)
            return x, y, None

        x, y = [], []
        with open(os.path.join(self.data_path, "GT-final_test.csv")) as f:
            next(f)  # skip header
            for line in f:
                line = line.strip().split(";")
                x.append(
                    os.path.join(self.data_path, "GTSRB", "Final_Test", "Images", line[0])
                )
                y.append(int(line[-1]))

        x = np.array(x)
        y = np.array(y)

        return x, y, None
