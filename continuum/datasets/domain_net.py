import os
from typing import Tuple

import numpy as np
from sklearn.model_selection import train_test_split
from torchvision import datasets as torchdata

from continuum.datasets import ImageFolderDataset
from continuum.download import download, unzip
from continuum.tasks import TaskType


class DomainNet(ImageFolderDataset):
    """DomainNet dataset group.

    Contain 6 different domains, each made of 345 classes.

    * Moment Matching for Multi-Source Domain Adaptation
      Peng et al.
      ICCV 2019
    """
    urls = [
        "http://csr.bu.edu/ftp/visda/2019/multi-source/groundtruth/clipart.zip",
        "http://csr.bu.edu/ftp/visda/2019/multi-source/infograph.zip",
        "http://csr.bu.edu/ftp/visda/2019/multi-source/groundtruth/painting.zip",
        "http://csr.bu.edu/ftp/visda/2019/multi-source/quickdraw.zip",
        "http://csr.bu.edu/ftp/visda/2019/multi-source/real.zip",
        "http://csr.bu.edu/ftp/visda/2019/multi-source/sketch.zip"
    ]

    def __init__(self, data_path, train: bool = True, download: bool = True,
                 test_split: float = 0.2, random_seed: int = 1):
        self.test_split = test_split
        self.random_seed = random_seed
        super().__init__(data_path, train, download)

    @property
    def data_type(self) -> TaskType:
        return TaskType.IMAGE_PATH

    def _download(self):
        for url in self.urls:
            base_file_name = url.split("/")[-1].split(".")[-2]

            path = os.path.join(self.data_path, base_file_name)
            if not os.path.exists(path):
                zip_path = path + ".zip"

                if not os.path.exists(zip_path):
                    print(f"Downloading {base_file_name}...")
                    download(url, self.data_path)

                print('Extracting archive...', end=' ')
                unzip(zip_path)
                print('Done!')

    def get_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        domains = ["clipart", "infograph", "painting", "quickdraw", "real", "sketch"]

        full_x, full_y, full_t = [], [], []

        for domain_id, domain_name in enumerate(domains):
            dataset = torchdata.ImageFolder(os.path.join(self.data_path, domain_name))

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
