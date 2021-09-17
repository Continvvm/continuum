import os
import json
import multiprocessing as mp
from typing import Tuple

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from torchvision import datasets as torchdata

from continuum.datasets import ImageFolderDataset
from continuum.download import download, untar, unzip


class TerraIncognita(ImageFolderDataset):
    """TerraIncognita dataset group.

    Filtered according to DomainBed rule, whose code was largely used here:
    https://github.com/facebookresearch/DomainBed/blob/main/domainbed/scripts/download.py#L167

    Contain 4 different domains.
    Each made of 10 animal classes.

    * Recognition in Terra Incognita
      Beery et al.
      ECCV 2018
    """
    images_url = "https://lilablobssc.blob.core.windows.net/caltechcameratraps/eccv_18_all_images_sm.tar.gz"
    json_url = "https://lilablobssc.blob.core.windows.net/caltechcameratraps/labels/caltech_camera_traps.json.zip"

    def __init__(self, data_path, train: bool = True, download: bool = True,
                 test_split: float = 0.2, random_seed: int = 1):
        self._attributes = None
        self.test_split = test_split
        self.random_seed = random_seed
        super().__init__(data_path, train, download)

    @property
    def data_type(self):
        return "image_data_path"

    def _download(self):
        if not os.path.exists(os.path.join(self.data_path, "eccv_18_all_images_sm")):
            tar_path = os.path.join(self.data_path, "eccv_18_all_images_sm.tar.gz")
            if not os.path.exists(tar_path):
                print("Downloading images archive...", end=" ")
                download(self.images_url, self.data_path)
                print("Done!")
            print('Extracting archive...', end=' ')
            untar(zip_path)
            print('Done!')

        if not os.path.exists(os.path.join(self.data_path, "caltech_camera_traps.json")):
            zip_path = os.path.join(self.data_path, "caltech_camera_traps.json.zip")
            if not os.path.exists(zip_path):
                print("Downloading json archive...", end=" ")
                download(self.json_url, self.data_path)
                print("Done!")
            print('Extracting archive...', end=' ')
            unzip(zip_path)
            print('Done!')

    def get_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        path_x = os.path.join(self.data_path, "continuum_terrainc_x.npy")
        path_y = os.path.join(self.data_path, "continuum_terrainc_y.npy")
        path_t = os.path.join(self.data_path, "continuum_terrainc_t.npy")

        if not all(os.path.exist(p) for p in [path_x, path_y, path_t]):
            print("Long (~1min) preprocessing starting! It will be cached for next time.")
            x, y, t = self._preprocess_data()
            np.save(x, path_x)
            np.save(y, path_y)
            np.save(t, path_t)
        else:
            x = np.load(path_x)
            y = np.load(path_y)
            t = np.load(path_t)

        return x, y, t

    def _preprocess_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Quite long (~1min) preprocessing as done per DomainBed github.

        The results of the preprocessing is saved in the same folder so it can
        be avoid on second time.

        See:
        - https://github.com/facebookresearch/DomainBed/blob/main/domainbed/scripts/download.py#L167
        """
        include_locations = ["38", "46", "100", "43"]
        include_categories = [
            "bird", "bobcat", "cat", "coyote", "dog", "empty", "opossum", "rabbit",
            "raccoon", "squirrel"
        ]

        images_folder = os.path.join(self.data_path, "eccv_18_all_images_sm/")
        annotations_file = os.path.join(self.data_path, "caltech_images_20210113.json")

        x, y, t = [], [], []
        with open(annotations_file, "r") as f:
            data = json.load(f)

        category_dict = {}
        for item in data['categories']:
            category_dict[item['id']] = item['name']

        def _func(image):
            image_location = image['location']

            if image_location not in include_locations:
                return None

            image_id = image['id']
            image_fname = image['file_name']

            x, y, t = [], [], []
            for annotation in data['annotations']:
                if annotation['image_id'] == image_id:
                    category = category_dict[annotation['category_id']]

                    if category not in include_categories:
                        return None

                    x.append(os.path.join(images_folder, image_id + ".jpg"))
                    y.append(include_categories.index(category))
                    t.append(include_locations.index(image_location))

            return x, y, t

        x, y, t = [], [], []
        with mp.Pool(min(8, mp.cpu_count())) as pool:
            for tup in pool.imap(_func, data['images']):
                if tup is None:
                    continue
                x.extend(tup[0])
                y.extend(tup[1])
                t.extend(tup[2])

        assert len(x) == len(y) == len(t)

        x, y, t = np.concatenate(x), np.concatenate(y), np.concatenate(t)
        return x, y, t
