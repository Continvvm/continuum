import os
from typing import Tuple, Union

from torchvision import datasets as torchdata
import numpy as np

from continuum.datasets import ImageFolderDataset
from continuum.download import download, unzip


class ImageNet1000(ImageFolderDataset):
    """ImageNet1000 dataset.

    Simple wrapper around ImageFolderDataset to provide a link to the download
    page.
    """

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


class TinyImageNet200(ImageFolderDataset):
    """Smaller version of ImageNet.

    - 200 classes
    - 500 images per class
    - size 64x64
    """

    url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"

    def _download(self):
        path = os.path.join(self.data_path, "tiny-imagenet-200")
        if not os.path.exists(f"{path}.zip"):
            download(self.url, self.data_path)
        if not os.path.exists(path):
            unzip(f"{path}.zip")

        print("TinyImagenet is downloaded.")

    def get_data(self) -> Tuple[np.ndarray, np.ndarray, Union[np.ndarray, None]]:
        return self._format(
            torchdata.ImageFolder(
                os.path.join(self.data_path, "tiny-imagenet-200", "train" if self.train else "val")
            ).imgs
        )
