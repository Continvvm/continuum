import os
from typing import List

import numpy as np
from torchvision import datasets as torchdata

from continuum.datasets import ImageFolderDataset
from continuum import download
from continuum.tasks import TaskType


class DTD(ImageFolderDataset):
    """Describable Textures Dataset (DTD)

    Reference:
        * Describing Textures in the Wild
          M. Cimpoi and S. Maji and I. Kokkinos and S. Mohamed and and A. Vedaldi
          CVPR 2014
    """
    url = "https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz"

    def __init__(self, data_path: str, train: bool = True, download: bool = True, split: int = 1):
        super().__init__(data_path=data_path, train=train, download=download, data_type=TaskType.IMAGE_PATH)

        if not (1 <= int(split) <= 10):
            raise ValueError(f"Available splits are [1, ..., 10], not {split}")
        self.split = split

    def _download(self):
        archive_path = os.path.join(self.data_path, "dtd-r1.0.1.tar.gz")
        if not os.path.exists(archive_path):
            print("Downloading DTD dataset...")
            download.download(self.url, self.data_path)
        if not os.path.exists(os.path.join(self.data_path, "dtd")):
            print("Uncompressing images...")
            download.untar(archive_path)

    def get_data(self):
        x, y, t = self._format(torchdata.ImageFolder(os.path.join(self.data_path, "dtd", "images")).imgs)

        if self.train:
            index_files = [
                os.path.join(self.data_path, "dtd", "labels", f"train{str(self.split)}.txt"),
                os.path.join(self.data_path, "dtd", "labels", f"val{str(self.split)}.txt")
            ]
        else:
            index_files = [
                os.path.join(self.data_path, "dtd", "labels", f"test{str(self.split)}.txt")
            ]

        valid_paths = set()
        for index_file in index_files:
            with open(index_file) as f:
                valid_paths.update(
                    map(lambda p: os.path.join(self.data_path, "dtd", "images", p.strip()),
                        f.readlines()
                    )
                )
        valid_paths = np.array(list(valid_paths))
        indexes = np.isin(x, valid_paths)

        return x[indexes], y[indexes], None

