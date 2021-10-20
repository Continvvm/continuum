import os
from typing import Tuple

import numpy as np

from continuum.datasets import _ContinuumDataset
from continuum.tasks import TaskType
from continuum.download import untar


class FER2013(_ContinuumDataset):
    """Facial Expression Recognition Challenge 2014 dataset.

    Grayscale images with 7 categories of facial emotions.

    * Kaggle https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/
    """

    def __init__(self, data_path, train: bool = True, download: bool = False):
        super().__init__(data_path, train, download)

    @property
    def data_type(self) -> TaskType:
        return TaskType.IMAGE_ARRAY

    def _download(self):
        archive_path = os.path.join(self.data_path, "fer2013.tar.gz")

        if not os.path.exists(archive_path):
            raise Exception(
                "You need to download this dataset yourself at "
                "https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data?select=fer2013.tar.gz"
            )
        if not os.path.exists(os.path.join(self.data_path, "fer2013")):
            print("Extracting archive...", end=" ")
            untar(archive_path)
            print("Done!")

    def get_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        x, y = [], []

        with open(os.path.join(self.data_path, "fer2013", "fer2013.csv")) as f:
            next(f)  # skip header
            for line in f:
                emotion, pixels, training = line.strip().split(",")
                if (self.train and training != "training") or (not self.train and training == "training"):
                    continue

                y.append(int(emotion))
                pixels = np.array(list(map(int, pixels.split(" ")))).reshape(48, 48).astype(np.uint8)
                x.append(pixels)

        x = np.stack(x)
        y = np.array(y)
        return x, y, None
