import os
import glob
from typing import Tuple

import numpy as np
from sklearn.model_selection import train_test_split

from continuum.datasets import _ContinuumDataset
from continuum.download import download, untar
from continuum.tasks import TaskType


class SUN397(_ContinuumDataset):
    """SUN397 dataset.

    Dataset with large images (usually > 1000px wide) and 397 classes.

    * SUN Database: Large-scale Scene Recognition from Abbey to Zoo.
      J. Xiao, J. Hays, K. Ehinger, A. Oliva, and A. Torralba.
      CVPR 2010
    """
    images_url = "http://vision.princeton.edu/projects/2010/SUN/SUN397.tar.gz"

    def __init__(self, data_path, train: bool = True, download: bool = True, test_split: float = 0.2,
                 random_seed=1):
        self.test_split = test_split
        self.random_seed = random_seed
        super().__init__(data_path, train, download)

    @property
    def data_type(self) -> TaskType:
        return TaskType.IMAGE_PATH

    def _download(self):
        if not os.path.exists(os.path.join(self.data_path, "SUN397")):
            archive_path = os.path.join(self.data_path, "SUN397.tar.gz")

            if not os.path.exists(archive_path):
                print("Downloading images archive...", end=' ')
                download(self.images_url, self.data_path)
                print('Done!')

            print('Extracting archive...', end=' ')
            untar(archive_path)
            print('Done!')


    def get_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        x, y = [], []

        with open(os.path.join(self.data_path, "SUN397", "ClassName.txt")) as f:
            for class_id, line in enumerate(f):
                line = line.strip()[1:]
                paths = glob.glob(os.path.join(self.data_path, "SUN397", line, "*.jpg"))

                x.extend(paths)
                y.extend([class_id for _ in range(len(paths))])

        x = np.array(x)
        y = np.array(y)

        x_train, x_test, y_train, y_test = train_test_split(
            x, y,
            test_size=self.test_split,
            random_state=self.random_seed
        )

        if self.train:
            return x_train, y_train, None
        return x_test, y_test, None
