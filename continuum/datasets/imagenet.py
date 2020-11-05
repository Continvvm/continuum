import os
from typing import Tuple, Union

import numpy as np

from continuum.datasets import ImageFolderDataset
from continuum.download import download


class ImageNet1000(ImageFolderDataset):
    """ImageNet1000 dataset.

    Simple wrapper around ImageFolderDataset to provide a link to the download
    page.
    """

    def _download(self):
        """
        Downloads the folder.

        Args:
            self: (todo): write your description
        """
        if not os.path.exists(self.data_folder):
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
        """
        Initialize data array.

        Args:
            self: (todo): write your description
            data_subset: (str): write your description
            Union: (todo): write your description
            Tuple: (todo): write your description
            np: (int): write your description
            array: (array): write your description
            np: (int): write your description
            array: (array): write your description
            str: (todo): write your description
        """
        self.data_subset = data_subset
        super().__init__(*args, **kwargs)

    def _download(self):
        """
        Downloads the dataset.

        Args:
            self: (todo): write your description
        """
        super()._download()

        filename = "val_100.txt"
        self.subset_url = self.test_subset_url
        if self.train:
            filename = "train_100.txt"
            self.subset_url = self.train_subset_url

        if self.data_subset is None:
            self.data_subset = os.path.join(self.data_folder, filename)
            download(self.subset_url, self.data_folder)

    def get_data(self) -> Tuple[np.ndarray, np.ndarray, Union[np.ndarray, None]]:
        """
        Get the data from the dataset.

        Args:
            self: (todo): write your description
        """
        data = self._parse_subset(self.data_subset, train=self.train)  # type: ignore
        return (*data, None)

    def _parse_subset(
        self,
        subset: Union[Tuple[np.array, np.array], str, None],
        train: bool = True
    ) -> Tuple[np.array, np.array]:
        """
        Reads a subset of the dataset.

        Args:
            self: (todo): write your description
            subset: (todo): write your description
            Union: (str): write your description
            Tuple: (str): write your description
            np: (todo): write your description
            array: (array): write your description
            np: (todo): write your description
            array: (array): write your description
            str: (str): write your description
            train: (bool): write your description
        """
        if isinstance(subset, str):
            x, y = [], []

            with open(subset, "r") as f:
                for line in f:
                    split_line = line.split(" ")
                    path = "/".join(split_line[0].strip().split("/")[1:])
                    x.append(os.path.join(self.data_folder, path))
                    y.append(int(split_line[1].strip()))
            x = np.array(x)
            y = np.array(y)
            return x, y
        return subset  # type: ignore
