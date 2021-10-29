import os
from typing import Tuple, Union, Optional

from torchvision import transforms
import numpy as np
from continuum.tasks import TaskType

from continuum.datasets import ImageFolderDataset, _ContinuumDataset
from continuum.download import download, unzip


class ImageNet1000(ImageFolderDataset):
    """ImageNet1000 dataset.

    Simple wrapper around ImageFolderDataset to provide a link to the download
    page.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.train:
            self.data_path = os.path.join(self.data_path, "train")
        else:
            self.data_path = os.path.join(self.data_path, "val")

    @property
    def transformations(self):
        """Default transformations if nothing is provided to the scenario."""
        return [transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]

    def _download(self):
        if not os.path.exists(self.data_path):
            raise IOError(
                "You must download yourself the ImageNet dataset."
                " Please go to http://www.image-net.org/challenges/LSVRC/2012/downloads and"
                " download 'Training images (Task 1 & 2)' and 'Validation images (all tasks)'."
            )
        print("ImageNet already downloaded.")

class ImageNet100(ImageNet1000):
    """Subset of ImageNet1000 made of only 100 classes.

    You must download the ImageNet1000 dataset then provide the images subset.
    If in doubt, use the option at initialization `download=True` and it will
    auto-download for you the subset ids used in:
        * Small Task Incremental Learning
          Douillard et al. 2020
    """

    train_subset_url = "https://github.com/Continvvm/continuum/releases/download/v0.1/train_100.txt"
    test_subset_url = "https://github.com/Continvvm/continuum/releases/download/v0.1/val_100.txt"

    def __init__(
            self, *args, data_subset: Union[Tuple[np.array, np.array], str, None] = None, **kwargs
    ):
        self.data_subset = data_subset
        super().__init__(*args, **kwargs)

    def _download(self):
        super()._download()

        filename = "val_100.txt"
        self.subset_url = self.test_subset_url
        if self.train:
            filename = "train_100.txt"
            self.subset_url = self.train_subset_url

        if self.data_subset is None:
            self.data_subset = os.path.join(self.data_path, filename)
            download(self.subset_url, self.data_path)

    def get_data(self) -> Tuple[np.ndarray, np.ndarray, Union[np.ndarray, None]]:
        data = self._parse_subset(self.data_subset, train=self.train)  # type: ignore
        return (*data, None)

    def _parse_subset(
            self,
            subset: Union[Tuple[np.array, np.array], str, None],
            train: bool = True
    ) -> Tuple[np.array, np.array]:
        if isinstance(subset, str):
            x, y = [], []

            with open(subset, "r") as f:
                for line in f:
                    split_line = line.split(" ")
                    path = split_line[0].strip()
                    x.append(os.path.join(self.data_path, path))
                    y.append(int(split_line[1].strip()))
            x = np.array(x)
            y = np.array(y)
            return x, y
        return subset  # type: ignore


class TinyImageNet200(_ContinuumDataset):
    """Smaller version of ImageNet.

    - 200 classes
    - 500 images per class
    - size 64x64
    """

    url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
    num_classes = 200

    def _download(self):
        path = os.path.join(self.data_path, "tiny-imagenet-200")
        if not os.path.exists(path):
            if not os.path.exists(f"{path}.zip"):
                download(self.url, self.data_path)
            unzip(f"{path}.zip")

    @property
    def data_type(self) -> TaskType:
        return TaskType.IMAGE_PATH

    def get_data(self) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        # First load wnids
        wnids_file = os.path.join(self.data_path, "tiny-imagenet-200","wnids.txt")
        with open(os.path.join(wnids_file), "r") as f:
            wnids = [x.strip() for x in f]

        # Map wnids to integer labels
        wnid_to_label = {wnid: i for i, wnid in enumerate(wnids)}

        if not self.train:
            # Next load validation data
            val_files = []
            val_wnids = []
            with open(os.path.join(self.data_path, "tiny-imagenet-200", "val", "val_annotations.txt"), "r") as f:
                for line in f:
                    # Select only validation images in chosen wnids set
                    if line.split()[1] in wnids:
                        img_file, wnid = line.split("\t")[:2]
                        val_files.append(os.path.join(self.data_path, "tiny-imagenet-200", "val", "images", img_file))
                        val_wnids.append(wnid)
            x_val = np.array(val_files)
            y_val = np.array([wnid_to_label[wnid] for wnid in val_wnids])
            return x_val, y_val, None

        # Next load training data.
        x_train = []
        y_train = []
        for wnid in wnids:
            # To figure out the filenames we need to open the boxes file
            boxes_file = os.path.join(self.data_path, "tiny-imagenet-200", "train", wnid, "%s_boxes.txt" % wnid)
            with open(boxes_file, "r") as f:
                train_filenames = [
                    os.path.join(self.data_path, "tiny-imagenet-200", "train", wnid, "images", x.split("\t")[0])
                    for x in f
                ]
            num_images = len(train_filenames)

            x_train.append(train_filenames)
            y_train_block = wnid_to_label[wnid] * np.ones(num_images, dtype=np.int64)
            y_train.append(y_train_block)

        # We need to concatenate all training data
        x_train = np.concatenate(x_train, axis=0)
        y_train = np.concatenate(y_train, axis=0)

        return x_train, y_train, None
