import os
from typing import Iterable, Set, Tuple, Union

import numpy as np

from continuum import download
from continuum.datasets.base import _ContinuumDataset


class _SemanticSegmentationDataset(_ContinuumDataset):
    @property
    def data_type(self) -> str:
        return "semantic_segmentation"


class PascalVOC2012(_SemanticSegmentationDataset):
    """PascalVOC2012 Semantic Segmentation Dataset.

    :param data_path: Path where the data is present or will be downloaded.
    :param download: Whether to download.
    :param augmented: Use augmented version of PascalVOC (recommended).
    """

    data_url = "http://pjreddie.com/media/files/VOCtrainval_11-May-2012.tar"
    segmentation_url = "http://cs.jhu.edu/~cxliu/data/SegmentationClassAug.zip"
    split_url = "http://cs.jhu.edu/~cxliu/data/list.zip"

    def __init__(self, data_path: str = "", download: bool = True, augmented: bool = True) -> None:
        super().__init__(data_path, download)
        self.augmented = augmented

    def _download(self):
        # Downloading images
        if not os.path.exists(os.path.join(self.data_path, "VOCdevkit")):
            path = os.path.join(self.data_path, "VOCtrainval_11-May-2012.tar")
            if not os.path.exists(path):
                print("Downloading Pascal VOC segmentation maps...")
                download.download(self.data_url, self.data_path)
            print("Uncompressing images...")
            download.untar(path)

        # Downloading segmentation maps
        if not os.path.exists(os.path.join(self.data_path, "SegmentationClassAug")):
            path = os.path.join(self.data_path, "SegmentationClassAug.zip")
            if not os.path.exists(path):
                print("Downloading Pascal VOC segmentation maps...")
                download.download(self.segmentation_url, self.data_path)
            print("Uncompressing segmentation maps...")
            download.unzip(path)

        # Downloading train/val/test indexes
        if not os.path.exists(os.path.join(self.data_path, "list")):
            path = os.path.join(self.data_path, "list.zip")
            if not os.path.exists(path):
                print("Downloading Pascal VOC train/val/test indexes...")
                download.download(self.split_url, self.data_path)
            print("Uncompressing train/val/test indexes...")
            download.unzip(path)

    def get_data(self, train: bool) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if train and self.augmented:
            list_path = os.path.join(self.data_path, "list", "train_aug.txt")
        elif train:
            list_path = os.path.join(self.data_path, "list", "train.txt")
        else:
            list_path = os.path.join(self.data_path, "list", "val.txt")

        image_paths, map_paths = [], []
        with open(list_path, "r") as f:
            for line in f:
                p1, p2 = line.split(" ")
                image_paths.append(os.path.join(self.data_path, "VOCdevkit", "VOC2012", p1))
                map_paths.append(os.path.join(self.data_path, p2))

        return np.array(image_paths), np.array(map_paths), None
