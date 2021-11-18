import os
from typing import Tuple

import numpy as np
from scipy import io

from continuum.datasets import _ContinuumDataset
from continuum.download import download, untar
from continuum.tasks import TaskType


class Car196(_ContinuumDataset):
    """Car196 dataset.

    * 3D Object Representations for Fine-Grained Categorization
      Jonathan Krause, Michael Stark, Jia Deng, Li Fei-Fei
      ICCV 2013 Workshop
    """
    devkit_url = "http://ai.stanford.edu/~jkrause/cars/car_devkit.tgz"
    train_url = "http://ai.stanford.edu/~jkrause/car196/cars_train.tgz"
    test_url = "http://ai.stanford.edu/~jkrause/car196/cars_test.tgz"
    test_labels_url = "http://ai.stanford.edu/~jkrause/car196/cars_test_annos_withlabels.mat"

    def __init__(self, data_path, train: bool = True, download: bool = True, mode: str = "classification"):
        super().__init__(data_path, train, download)

        if mode not in ("classification", "detection"):
            raise ValueError(f"Unsupported mode <{mode}>, available are <classification> and <detection>.")
        if mode == "detection":
            raise NotImplementedError("Detection is not yet supported by Continuum, sorry!")
        self.mode = mode

    @property
    def data_type(self) -> TaskType:
        return TaskType.IMAGE_PATH

    def _download(self):
        folders = ["devkit", "cars_train", "cars_test"]
        archives = ["car_devkit.tgz", "cars_train.tgz", "cars_test.tgz"]
        urls = [self.devkit_url, self.train_url, self.test_url]

        for f, a, u in zip(folders, archives, urls):
            if not os.path.exists(os.path.join(self.data_path, f)):
                archive_path = os.path.join(self.data_path, a)

                if not os.path.exists(archive_path):
                    print(f"Downloading archive {a} ...", end=" ")
                    download(u, self.data_path)
                    print('Done!')

                print(f"Extracting archive... {a}->{f}", end=" ")
                untar(archive_path)
                print("Done!")

        if not os.path.exists(os.path.join(self.data_path, "cars_test_annos_withlabels.mat")):
            download(self.test_labels_url, self.data_path)

    def get_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.train:
            mat = io.loadmat(os.path.join(self.data_path, "devkit", "cars_train_annos.mat"))
            folder = "cars_train"
        else:
            mat = io.loadmat(os.path.join(self.data_path, "cars_test_annos_withlabels.mat"))
            folder = "cars_test"
        mat = mat["annotations"][0]

        x, y = [], []
        for index in range(len(mat)):
            # x1, x2, y1, y2 at indexes 0, 1, 2, 3
            class_id = mat[index][4].item()
            image_id = mat[index][5].item()
            x.append(os.path.join(self.data_path, folder, image_id))
            y.append(class_id)

        return np.array(x), np.array(y) - 1, None
