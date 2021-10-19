from datetime import date
import os
import warnings
from multiprocessing.pool import ThreadPool
from typing import Optional, Tuple
import hashlib

import numpy as np
import requests

from continuum.datasets import _ContinuumDataset
from continuum.download import ProgressBar, download, untar
from continuum.tasks import TaskType

_DOWNLOAD_FOLDER = None


class Birdsnap(_ContinuumDataset):
    """Birdsnap dataset.

    500 classes of birds.

    Warnings, this dataset downloads a list of images from Flickr. Some images
    will be missing. Furthermore the whole dataset is quite big on disk (~60go).

    * Birdsnap: Large-scale Fine-grained Visual Categorization of Birds
      T. Berg, J. Liu, S. W. Lee, M. L. Alexander, D. W. Jacobs, and P. N. Belhumeur
      CVPR 2014

    :param crop_bbox: Use the provided bounding boxes for maximal cropping around the birds.
    """
    meta_url = "http://thomasberg.org/datasets/birdsnap/1.1/birdsnap.tgz"

    def __init__(self, data_path, train: bool = True, download: bool = True, crop_bbox: bool = False):
        self.crop_bbox = crop_bbox
        self._bboxes = None

        super().__init__(data_path, train, download)

    @property
    def data_type(self) -> TaskType:
        return TaskType.IMAGE_PATH

    @property
    def bounding_boxes(self) -> Optional[np.ndarray]:
        if self.crop_bbox:
            return self._bboxes
        return None

    def _download(self):
        global _DOWNLOAD_FOLDER
        _DOWNLOAD_FOLDER = os.path.join(self.data_path, "images")

        os.makedirs(_DOWNLOAD_FOLDER, exist_ok=True)

        if not os.path.exists(os.path.join(self.data_path, "birdsnap")):
            archive_path = os.path.join(self.data_path, "birdsnap.tgz")

            if not os.path.exists(archive_path):
                print("Downloading archive of metadata...", end=' ')
                download(self.meta_url, self.data_path)
                print('Done!')

            print('Extracting archive...', end=' ')
            untar(archive_path)
            print('Done!')

        with open(os.path.join(self.data_path, "birdsnap", "images.txt")) as f:
            data = f.readlines()[1:]

        good_images = 0
        print(f"Downloading or checking {len(data)} images...")
        pb = ProgressBar()
        with ThreadPool(10) as pool:
            for processed_data in pool.imap_unordered(_download_images, data):
                pb.update(None, 1, len(data))

                if processed_data is None:
                    continue
                good_images += 1

        pb.end(len(data))
        if good_images != len(data):
            warnings.warn(f"{len(data)-good_images} couldn't be downloaded among {len(data)}.")

    def get_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        x, y, bboxes = [], [], []

        with open(os.path.join(self.data_path, "birdsnap", "test_images.txt")) as f:
            next(f)  # skip header
            test_paths = set(map(lambda x: x.strip(), f.readlines()))

        with open(os.path.join(self.data_path, "birdsnap", "images.txt")) as f:
            next(f)  # skip header
            for line in f:
                line = list(filter(lambda x: len(x) > 0, line.split("\t")))
                path = line[2]
                class_id = line[3]
                x1, y1, x2, y2 = line[4:8]

                full_path = os.path.join(self.data_path, "images", path)
                if not os.path.exists(full_path):
                    continue

                if (self.train and path in test_paths) or (not self.train and path not in test_paths):
                    continue

                x.append(full_path)
                y.append(class_id)
                bboxes.append([x1, y1, x2, y2])

        x = np.array(x)
        y = np.array(y, dtype=np.int64) - 1
        bboxes = np.array(bboxes, dtype=np.int32)

        self._bboxes = bboxes

        return x, y, None


def _download_images(line):
    global _DOWNLOAD_FOLDER

    line = list(filter(lambda x: len(x) > 0, line.split("\t")))
    url = line[0]
    md5 = line[1]
    path = line[2]
    class_id = line[3]
    x1, y1, x2, y2 = line[4:8]

    full_path = os.path.join(_DOWNLOAD_FOLDER, path)
    os.makedirs(os.path.dirname(full_path), exist_ok=True)

    if _check_image(md5, full_path):
        return (path, class_id, (x1, y1, x2, y2))

    try:
        r = requests.get(url)
        if r.status_code == 200:
                with open(full_path, 'wb') as fout:
                    for chunk in r.iter_content(1024):
                        fout.write(chunk)
        else:
            return None
    except:
        _clean_if_failed(full_path)
        return None

    if not _check_image(md5, full_path):
        return None

    return (path, class_id, (x1, y1, x2, y2))


def _check_image(md5, imagepath):
    if not os.path.exists(imagepath): return False
    with open(imagepath, 'rb') as fin:
        valid = hashlib.md5(fin.read()).hexdigest() == md5

    if not valid:
        _clean_if_failed(imagepath)
    return valid


def _clean_if_failed(imagepath):
    try:
        os.remove(imagepath)
    except:
        pass
