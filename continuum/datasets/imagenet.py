import os
from typing import Tuple, Union

from torchvision import datasets as torchdata
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
        if not os.path.exists(f"{path}.zip"):
            download(self.url, self.data_path)
        if not os.path.exists(path):
            unzip(f"{path}.zip")

        print("TinyImagenet is downloaded.")

    def get_classes_names(self):
        # First load wnids
        wnids_file = os.path.join(self.data_path, 'wnids.txt')
        with open(os.path.join(self.data_path, wnids_file), 'r') as f:
            wnids = [x.strip() for x in f]

        # Map wnids to integer labels
        wnid_to_label = {wnid: i for i, wnid in enumerate(wnids)}

        # Use words.txt to get names for each class
        words_file = os.path.join(self.data_path, 'words.txt')
        with open(os.path.join(self.data_path, words_file), 'r') as f:
            wnid_to_words = dict(line.split('\t') for line in f)
            for wnid, words in wnid_to_words.items():
                wnid_to_words[wnid] = [w.strip() for w in words.split(',')]

        class_names = [wnid_to_words[wnid] for wnid in wnids]
        return class_names

    @property
    def data_type(self) -> TaskType:
        return TaskType.IMAGE_PATH

    def get_data(self):
        """
        Code inspired from https://github.com/rmccorm4/Tiny-Imagenet-200
        Load TinyImageNet.

        Inputs:
        - path: String giving path to the directory to load.
        - dtype: numpy datatype used to load the data.

        Returns: A tuple of
        - class_names: A list where class_names[i] is a list of strings giving the
          WordNet names for class i in the loaded dataset.
        - X_train: (N_tr, 3, 64, 64) array of training images
        - y_train: (N_tr,) array of training labels
        - X_val: (N_val, 3, 64, 64) array of validation images
        - y_val: (N_val,) array of validation labels
        - X_test: (N_test, 3, 64, 64) array of testing images.
        - y_test: (N_test,) array of test labels; if test labels are not available
          (such as in student code) then y_test will be None.
        """
        # First load wnids
        wnids_file = os.path.join(self.data_path, 'tiny-imagenet-200','wnids.txt')
        with open(os.path.join(wnids_file), 'r') as f:
            wnids = [x.strip() for x in f]

        # Map wnids to integer labels
        wnid_to_label = {wnid: i for i, wnid in enumerate(wnids)}

        # Use words.txt to get names for each class
        words_file = os.path.join(self.data_path, 'tiny-imagenet-200', 'words.txt')
        with open(os.path.join(words_file), 'r') as f:
            wnid_to_words = dict(line.split('\t') for line in f)
            for wnid, words in wnid_to_words.items():
                wnid_to_words[wnid] = [w.strip() for w in words.split(',')]

        # Next load training data.
        X_train = []
        y_train = []
        for i, wnid in enumerate(wnids):
            if (i + 1) % 20 == 0:
                print('loading training data for synset %d / %d' % (i + 1, len(wnids)))
            # To figure out the filenames we need to open the boxes file
            boxes_file = os.path.join(self.data_path, 'tiny-imagenet-200', 'train', wnid, '%s_boxes.txt' % wnid)
            with open(boxes_file, 'r') as f:
                train_filenames = [os.path.join(self.data_path, 'tiny-imagenet-200', 'train', wnid, "images",x.split('\t')[0]) for x in f]
            num_images = len(train_filenames)

            X_train.append(train_filenames)
            y_train_block = wnid_to_label[wnid] * np.ones(num_images, dtype=np.int64)
            y_train.append(y_train_block)

        # We need to concatenate all training data
        X_train = np.concatenate(X_train, axis=0)
        y_train = np.concatenate(y_train, axis=0)

        # Next load validation data
        with open(os.path.join(self.data_path, 'tiny-imagenet-200', 'val', 'val_annotations.txt'), 'r') as f:
            val_files = []
            val_wnids = []
            for line in f:
                # Select only validation images in chosen wnids set
                if line.split()[1] in wnids:
                    img_file, wnid = line.split('\t')[:2]
                    val_files.append(os.path.join(self.data_path, 'tiny-imagenet-200', 'val', img_file))
                    val_wnids.append(wnid)
            y_val = np.array([wnid_to_label[wnid] for wnid in val_wnids])

        if self.train:
            return X_train, y_train, None
        return val_files, y_val, None
