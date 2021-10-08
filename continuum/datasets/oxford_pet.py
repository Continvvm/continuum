import os
from typing import Tuple

import numpy as np

from continuum.datasets import _ContinuumDataset
from continuum.download import download, untar
from continuum.tasks import TaskType

class OxfordPet(_ContinuumDataset):
    """Oxford-IIIT Pet Dataset

      This is a 37 category pet dataset with roughly 200 images for each class. 
      The images have a large variations in scale, pose and lighting.
      All images have an associated ground truth annotation of breed.

    """
    base_url = "http://www.robots.ox.ac.uk/~vgg/data/pets/data"

    def __init__(self, data_path, train: bool = True, download: bool = True):
        self._attributes = None
        super().__init__(data_path, train, download)

    @property
    def data_type(self) -> TaskType:
        return TaskType.IMAGE_PATH

    def _download(self):
        if not os.path.exists(os.path.join(self.data_path, "images")):
            archive_images_path = os.path.join(self.data_path, "images.tar.gz")

            if not os.path.exists(archive_images_path):
                print("Downloading images archive...", end=' ')
                image_url = os.path.join(self.base_url, "images.tar.gz")
                download(image_url, self.data_path)
                print('Done!')

            print('Extracting images archive...', end=' ')
            untar(archive_images_path)
            print('Done!')

        if not os.path.exists(os.path.join(self.data_path, "annotations")):
            archive_annotations_path = os.path.join(self.data_path, "annotations.tar.gz")

            if not os.path.exists(archive_annotations_path):
                print("Downloading annotations archive...", end=' ')
                annotations_url = os.path.join(self.base_url, "annotations.tar.gz")
                download(annotations_url, self.data_path)
                print('Done!')

            print('Extracting annotations archive...', end=' ')
            untar(archive_annotations_path)
            print('Done!')

    def get_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # find the split
        split_file_name = "trainval.txt" if self.train else "test.txt"
        split_file_path = os.path.join(self.data_path, "annotations", split_file_name)
        
        x = []
        y = []
        with open(split_file_path, 'r') as file:
            for line in file:
                image_name, label, _, _ = line.strip().split(" ")
                path = os.path.join(self.data_path, "images", image_name+".jpg")
                if os.path.exists(path):
                    x.append(path)
                    y.append(int(label) - 1)

        x, y = np.array(x), np.array(y)
        
        return x, y, None