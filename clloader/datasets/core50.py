import glob
import os
from typing import List, Tuple, Union

import numpy as np

from clloader.datasets.base import _ContinuumDataset


class ImageFolderDataset(_ContinuumDataset):

    def __init__(
        self, folder: str, train_image_ids: Union[str, List[str]], download: bool = True, **kwargs
    ):
        super().__init__(download=download, **kwargs)

        self.folder = folder
        self.train_image_ids = train_image_ids

        if download:
            self._download()

    @property
    def in_memory(self):
        return False

    def _download(self):
        raise ValueError("bouh")

    def init(self, train: bool) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        x, y, t = [], [], []

        train_images_ids = set()
        if isinstance(self.train_paths, str):
            with open(self.train_paths, "r") as f:
                next(f)
                for line in f:
                    image_id = line.split(",")[0].split(".")[0]
                    train_images_ids.add(image_id)
        else:
            train_images_ids = set(self.train_paths)

        for domain_id in range(10):
            domain_folder = os.path.join(self.train_folder, f"s{domain_id + 1}")

            for object_id in range(50):
                object_folder = os.path.join(domain_folder, f"o{object_id + 1}")

                for path in os.listdir(object_folder):
                    image_id = path.split(".")[0]

                    if train and image_id not in train_images_ids:
                        continue
                    elif not train and image_id in train_images_ids:
                        continue

                    x.append(os.path.join(object_folder, path))
                    y.append(object_id)
                    t.append(domain_id)

        x = np.concatenate(x)
        y = np.concatenate(y)
        t = np.concatenate(t)

        return x, y, t
