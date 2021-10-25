import os
from typing import Tuple, List

import pandas as pd
import numpy as np

from continuum.datasets.base import _ContinuumDataset
from continuum.download import download_file_from_google_drive, untar
from continuum.tasks import TaskType


class CUB200(_ContinuumDataset):
    # initial code taken from https://github.com/TDeVries/cub2011_dataset
    base_folder = "CUB_200_2011/images"

    def __init__(self, data_path, train: bool = True, download: bool = True):
        data_path = os.path.expanduser(data_path)
        self._attributes = None
        super().__init__(data_path, train, download)

    @property
    def attributes(self):
        if self._attributes is None:
            att = np.loadtxt(
                os.path.join(
                    self.data_path, "CUB_200_2011",
                    "attributes", "class_attribute_labels_continuous.txt"
                )
            )
            self._attributes = att / np.linalg.norm(att, axis=-1, keepdims=True)

        return self._attributes

    @property
    def data_type(self) -> TaskType:
        return TaskType.IMAGE_PATH

    def _download(self):
        if not os.path.exists(os.path.join(self.data_path, "CUB_200_2011")):
            tgz_path = os.path.join(self.data_path, "CUB_200_2011.tgz")

            if not os.path.exists(tgz_path):
                print("Downloading tgz archive...", end=' ')
                download_file_from_google_drive(
                    "1hbzc_P1FuxMkcabkgn9ZKinBwW683j45",
                    tgz_path
                )
                print('Done!')

            print('Extracting archive...', end=' ')
            untar(tgz_path)
            print('Done!')

    def _load_metadata(self):
        images = pd.read_csv(
            os.path.join(self.data_path, "CUB_200_2011", "images.txt"),
            sep=" ",
            names=["img_id", "filedata_path"])

        image_class_labels = pd.read_csv(
            os.path.join(self.data_path, "CUB_200_2011", "image_class_labels.txt"),
            sep=" ",
            names=["img_id", "target"])
        train_test_split = pd.read_csv(
            os.path.join(self.data_path, "CUB_200_2011", "train_test_split.txt"),
            sep=" ",
            names=["img_id", "is_training_img"])

        data = images.merge(image_class_labels, on="img_id")
        self.data = data.merge(train_test_split, on="img_id")

        if self.train:
            self.data = self.data[self.data.is_training_img == 1]
        else:
            self.data = self.data[self.data.is_training_img == 0]

        x = os.path.join(self.data_path, "CUB_200_2011", "images") + "/" + np.array(self.data["filedata_path"])
        y = np.array(self.data["target"]) - 1  # Targets start at 1 by default, so shift to 0

        self.dataset = [x, y, None]

    def _check_integrity(self):
        try:
            self._load_metadata()
        except Exception:
            return False

        for index, row in self.data.iterrows():
            filedata_path = os.path.join(self.data_path, self.base_folder, row.filedata_path)
            if not os.path.isfile(filedata_path):
                print(filedata_path)
                return False
        return True

    def __len__(self):
        """Len measures the number of data point from csv data."""
        return len(self.data)

    def get_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not self._check_integrity():
            raise RuntimeError(
                "Dataset not found or corrupted. Remove existing file and download again, "
                "or try to download the dataset manually "
                "at http://www.vision.caltech.edu/visipedia/CUB-200-2011.html")

        return self.dataset
