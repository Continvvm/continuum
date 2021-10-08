import os
from typing import Tuple

import numpy as np
import scipy.io as sio

from continuum.datasets import _ContinuumDataset
from continuum.download import download, untar
from continuum.tasks import TaskType

class OxfordFlower102(_ContinuumDataset):
    """Oxford Flowers 102

      The Oxford Flowers 102 dataset consists of 102 flower categories commonly occurring
      in the United Kingdom. Each class consists of between 40 and 258 images.
      The dataset is divided into a training set, a validation set and a test set.
      The training set and validation set each consist of 10 images per class (totalling 1020 images each).
      The test set consists of the remaining 6149 images (minimum 20 per class).

    """
    base_url = "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/"

    def __init__(self, data_path, train: bool = True, download: bool = True):
        self._attributes = None
        super().__init__(data_path, train, download)

    @property
    def data_type(self) -> TaskType:
        return TaskType.IMAGE_PATH

    def _download(self):
        if not os.path.exists(os.path.join(self.data_path, "jpg")):
            archive_images_path = os.path.join(self.data_path, "102flowers.tgz")

            if not os.path.exists(archive_images_path):
                print("Downloading images archive...", end=' ')
                image_url = os.path.join(self.base_url, "102flowers.tgz")
                download(image_url, self.data_path)
                print('Done!')

            print('Extracting archive...', end=' ')
            untar(archive_images_path)
            print('Done!')

        # Downloading label file
        if not os.path.exists(os.path.join(self.data_path, "imagelabels.mat")):
            label_url = os.path.join(self.base_url, "imagelabels.mat")
            download(label_url, self.data_path)

        # Downloading split file
        if not os.path.exists(os.path.join(self.data_path, "setid.mat")):
            split_url = os.path.join(self.base_url, "setid.mat")
            download(split_url, self.data_path)

    def get_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # load the labels data
        labels = sio.loadmat(os.path.join(self.data_path, "imagelabels.mat"))["labels"][0]

        # find the split
        split_name = "trnid" if self.train else "tstid"
        split_ids = sio.loadmat(os.path.join(self.data_path, "setid.mat"))[split_name][0]

        x, y = [], []
        for image_id in split_ids:
            file_name = "image_%05d.jpg" % image_id
            path = os.path.join(self.data_path, "jpg", file_name)
            if os.path.exists(path):
                x.append(path)
                y.append(labels[image_id-1] - 1)

        x, y = np.array(x), np.array(y)
        
        return x, y, None