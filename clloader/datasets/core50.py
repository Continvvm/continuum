import os
from typing import List, Tuple, Union

import numpy as np

from clloader.datasets.base import _ContinuumDataset


class CORe50(_ContinuumDataset):
    """Continuum version of the CORe50 dataset.

    References:
        * CORe50: a new Dataset and Benchmark for Continuous Object Recognition
          Lomonaco & Maltoni.
          CoRL 2017

    :param folder: The folder extracted from the official zip file.
    :param train_image_ids: The image ids belonging to the train set. Either the
                            containing them provided by the official webpage, or
                            a list of string.
    :param download: An option useless in this case.
    """

    data_url = "http://bias.csr.unibo.it/maltoni/download/core50/core50_128x128.zip"
    train_ids_url = "https://vlomonaco.github.io/core50/data/core50_train.csv"

    def __init__(
        self, folder: str, train_image_ids: Union[str, List[str]], download: bool = True, **kwargs
    ):
        super().__init__(download=download, **kwargs)

        self.folder = folder
        self.train_image_ids = train_image_ids

        if download:
            self._download()

    @property
    def data_type(self):
        return "image_path"

    def _download(self):
        if os.path.exists(self.folder):
            print("CORe50 already downloaded.")
        else:
            raise IOError(
                f"CORe50 was not found there: {self.folder}."
                f" Please download and unzip this: {self.data_url},"
                f" and get the train images ids there: {self.train_ids_url}."
            )

    def init(self, train: bool) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate the CORe50 data.

        CORe50, in one of its many iterations, is made of 50 objects, each present
        in 10 different domains (in-door, street, garden, etc.).

        In class incremental (NC) setting, those domains won't matter.

        In instance incremental (NI) setting, the domains come one after the other,
        but all classes are present since the first task. Seven domains are allocated
        for the train set, while 3 domains are allocated for the test set.

        In the case of the test set, all domains have the "dummy" label of 0. The
        authors designed this dataset with a fixed test dataset in mind.
        """
        x, y, t = [], [], []

        train_images_ids = set()
        if isinstance(self.train_image_ids, str):
            with open(self.train_image_ids, "r") as f:
                next(f)
                for line in f:
                    image_id = line.split(",")[0].split(".")[0]
                    train_images_ids.add(image_id)
        else:
            train_images_ids = set(self.train_image_ids)

        domain_counter = 0
        for domain_id in range(10):
            # We walk through the 10 available domains.
            domain_folder = os.path.join(self.folder, "core50_128x128", f"s{domain_id + 1}")

            has_images = False
            for object_id in range(50):
                # We walk through the 50 available object categories.
                object_folder = os.path.join(domain_folder, f"o{object_id + 1}")

                for path in os.listdir(object_folder):
                    image_id = path.split(".")[0]

                    if (train and image_id not in train_images_ids) \
                       or (not train and image_id in train_images_ids):
                        continue

                    x.append(os.path.join(object_folder, path))
                    y.append(object_id)
                    if train:  # We add a new domain id for the train set.
                        t.append(domain_counter)
                    else:  # Test set is fixed, therefore we artificially give a unique domain.
                        t.append(0)

                    has_images = True
            if has_images:
                domain_counter += 1

        x = np.array(x)
        y = np.array(y)
        t = np.array(t)

        return x, y, t
