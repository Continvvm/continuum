import os
from typing import Iterable, Set, Tuple, Union

import numpy as np

from continuum import download
from continuum.datasets.base import _ContinuumDataset
from continuum.transforms.segmentation import ToTensor
from continuum.tasks import TaskType


class PascalVOC2012(_ContinuumDataset):
    """PascalVOC2012 Semantic Segmentation Dataset.

    :param data_path: Path where the data is present or will be downloaded.
    :param download: Whether to download.
    :param augmented: Use augmented version of PascalVOC with more images (recommended).
    """

    data_url = "http://pjreddie.com/media/files/VOCtrainval_11-May-2012.tar"
    segmentation_url = "http://cs.jhu.edu/~cxliu/data/SegmentationClassAug.zip"
    split_url = "http://cs.jhu.edu/~cxliu/data/list.zip"

    def __init__(self, data_path: str = "", train: bool = True, download: bool = True, augmented: bool = True) -> None:
        super().__init__(data_path=data_path, train=train, download=download)
        self.augmented = augmented

    @property
    def data_type(self) -> TaskType:
        return TaskType.SEGMENTATION

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

    def get_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.train and self.augmented:
            list_path = os.path.join(self.data_path, "list", "train_aug.txt")
        elif self.train:
            list_path = os.path.join(self.data_path, "list", "train.txt")
        else:
            list_path = os.path.join(self.data_path, "list", "val.txt")

        image_paths, map_paths = [], []
        with open(list_path, "r") as f:
            for line in f:
                p1, p2 = line.split(" ")
                image_paths.append(os.path.join(self.data_path, "VOCdevkit", "VOC2012", p1[1:].strip()))
                map_paths.append(os.path.join(self.data_path, p2[1:].strip()))

        return np.array(image_paths), np.array(map_paths), None


class PascalVOC2007(_ContinuumDataset):
    """Pascal VOC 2007.

    For now, it only supports classification, which is the most commonly used
    setting with this dataset.
    If you wish to do segmentation, you should consider usinng PascalVOC2012.
    """
    url = "http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar"


    def __init__(self, data_path, train: bool = True, download: bool = True, mode: str = "classification"):
        if mode not in ("classification", "detection", "segmentation"):
            raise ValueError(f"Unsupported target <{mode}>, available are "
                             " <classification>, <detection>, and <segmentation>.")
        if mode in ("detection", "segmentation"):
            raise NotImplementedError(f"This mode <{mode}> is not yet supported.")

        self.mode = mode

        super().__init__(data_path, train, download)

    @property
    def data_type(self) -> TaskType:
        if self.mode == "classification":
            return TaskType.IMAGE_PATH
        elif self.mode  == "segmentation":
            return TaskType.SEGMENTATION
        return TaskType.DETECTION

    def _download(self):
        if not os.path.exists(os.path.join(self.data_path, "VOCdevkit", "VOC2007")):
            archive_path = os.path.join(self.data_path, "VOCtrainval_06-Nov-2007.tar")

            if not os.path.exists(archive_path):
                print(f"Downloading archive ...", end=" ")
                download.download(self.url, self.data_path)
                print('Done!')

            print(f"Extracting archive...", end=" ")
            download.untar(archive_path)
            print("Done!")

    def get_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        classes = [
            "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car",
            "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike",
            "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"
        ]

        if self.train:
            suffix = "train"
        else:
            suffix = "val"

        x, y = [], []
        for class_id, class_name in enumerate(classes):
            path = os.path.join(
                self.data_path,
                "VOCdevkit", "VOC2007", "ImageSets", "Main",
                f"{class_name}_{suffix}.txt"
            )

            with open(path) as f:
                for line in f:
                    line = line.split(" ")
                    presence = line[-1].strip()
                    if presence != "1":
                        continue

                    image_id = line[0].strip()
                    x.append(
                        os.path.join(
                            self.data_path,
                            "VOCdevkit", "VOC2007", "JPEGImages",
                            f"{image_id}.jpg"
                        )
                    )
                    y.append(class_id)

        return np.array(x), np.array(y), None
