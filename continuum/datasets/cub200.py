import os
from typing import Tuple

import pandas as pd
import numpy as np

from continuum.datasets.base import _ContinuumDataset


class CUB200(_ContinuumDataset):
    # initial code taken from https://github.com/TDeVries/cub2011_dataset
    base_folder = 'CUB_200_2011/images'

    def __init__(self, root_dir, train=True, transform=None):
        self.root = os.path.expanduser(root_dir)
        self.transform = transform
        self.train = train

    @property
    def data_type(self):
        return "image_path"

    def _load_metadata(self):
        images = pd.read_csv(
            os.path.join(self.root, 'CUB_200_2011', 'images.txt'),
            sep=' ',
            names=['img_id', 'filepath'])

        image_class_labels = pd.read_csv(
            os.path.join(self.root, 'CUB_200_2011', 'image_class_labels.txt'),
            sep=' ',
            names=['img_id', 'target'])
        train_test_split = pd.read_csv(
            os.path.join(self.root, 'CUB_200_2011', 'train_test_split.txt'),
            sep=' ',
            names=['img_id', 'is_training_img'])

        data = images.merge(image_class_labels, on='img_id')
        self.data = data.merge(train_test_split, on='img_id')

        if self.train:
            self.data = self.data[self.data.is_training_img == 1]
        else:
            self.data = self.data[self.data.is_training_img == 0]

        x = os.path.join(self.root, 'CUB_200_2011', 'images') + "/" + np.array(data["filepath"])
        y = np.array(data["target"]) - 1  # Targets start at 1 by default, so shift to 0

        self.dataset = [x, y, None]

    def _check_integrity(self):
        try:
            self._load_metadata()
        except Exception:
            return False

        for index, row in self.data.iterrows():
            filepath = os.path.join(self.root, self.base_folder, row.filepath)
            if not os.path.isfile(filepath):
                print(filepath)
                return False
        return True

    def __len__(self):
        """Len measures the number of data point from csv data."""
        return len(self.data)

    def get_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

        if not self._check_integrity():
            raise RuntimeError(
                'Dataset not found or corrupted. You need to download the dataset manually'
                ' at http://www.vision.caltech.edu/visipedia/CUB-200-2011.html')

        return self.dataset
