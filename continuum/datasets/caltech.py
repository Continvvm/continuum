import os
import glob
from typing import Tuple, List

import numpy as np
from sklearn.model_selection import train_test_split

from continuum.datasets import _ContinuumDataset
from continuum.download import download_file_from_google_drive, untar
from continuum.tasks import TaskType


class Caltech101(_ContinuumDataset):
    """Caltech 101 Dataset.

    Use the argument "remove_bg_google" to remove the class "BACKGROUND_Google",
    which is kinda not a real class. Without it the dataset has 101 classes,
    but some papers seems to use it and thus have 102 classes.
    """

    # Google drive ids
    data_id = "137RyRjvTBkBiIfeYBNZBtViDHQ6_Ewsp"
    folder = "101_ObjectCategories"

    def __init__(
            self,
            data_path: str = "",
            train: bool = True,
            download: bool = True,
            test_split: float = 0.2,
            random_seed: int = 1,
            remove_bg_google: bool = True
    ):

        super().__init__(data_path=data_path, train=train, download=download)

        self.test_split = test_split
        self.random_seed = random_seed
        self.remove_bg_google = remove_bg_google

    @property
    def data_type(self) -> TaskType:
        return TaskType.IMAGE_PATH

    def _download(self):
        data_folder = os.path.join(self.data_path, self.folder)
        annotation_folder = os.path.join(self.data_path, "Annotations")

        if not os.path.exists(data_folder):
            if not os.path.exists(data_folder + ".tar.gz"):
                print("Downloading data archive...", end=" ")
                download_file_from_google_drive(
                    self.data_id,
                    data_folder + ".tar.gz"
                )
                print("Done!")

            print("Extracting data archive...", end=" ")
            untar(data_folder + ".tar.gz")
            print("Done!")

    def get_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        categories = sorted(os.listdir(os.path.join(self.data_path, self.folder)))
        if self.remove_bg_google and "BACKGROUND_Google" in categories:
            categories.remove("BACKGROUND_Google")  # this is not a real class, in Caltech101

        x, y = [], []
        for (i, c) in enumerate(categories):
            for path in glob.iglob(os.path.join(self.data_path, self.folder, c, "*_*.jpg")):
                x.append(path)
                y.append(c)

        x, y = np.array(x), np.array(y)

        x_train, x_test, y_train, y_test = train_test_split(
            x, y,
            test_size=self.test_split,
            random_state=self.random_seed
        )

        if self.train:
            self.list_classes = np.unique(y_train)
            return x_train, y_train, None
        self.list_classes = np.unique(y_test)
        return x_test, y_test, None


class Caltech256(Caltech101):
    """Caltech 256 Dataset.

    Has 257 classes actually because why not.
    """

    data_id = "1r6o0pSROcV1_VwT4oSjA2FBUSCWGuxLK"
    folder = "256_ObjectCategories"
