import os
import pandas as pd
from typing import Tuple
import numpy as np

from torchvision.datasets.folder import default_loader
from torchvision.datasets.utils import download_url
from continuum.datasets.base import _ContinuumDataset


class CUB200(_ContinuumDataset):
    #initial code taken from https://github.com/TDeVries/cub2011_dataset
    base_folder = 'CUB_200_2011/images'
    url = 'http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz'
    filename = 'CUB_200_2011.tgz'
    tgz_md5 = '97eceeb196236b17998738112f37df78'

    def __init__(self, data_path, train=True, transform=None, download=True):
        self.root = os.path.expanduser(data_path)
        self.transform = transform
        self.loader = default_loader
        self.train = train

        if download:
            self._download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')
    @property
    def data_type(self):
        return "image_path"

    def _load_metadata(self):
        images = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'images.txt'), sep=' ', names=['img_id', 'filepath'])

        image_class_labels = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'image_class_labels.txt'), sep=' ', names=['img_id', 'target'])
        train_test_split = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'train_test_split.txt'),
                                       sep=' ', names=['img_id', 'is_training_img'])

        data = images.merge(image_class_labels, on='img_id')
        self.data = data.merge(train_test_split, on='img_id')

        if self.train:
            self.data = self.data[self.data.is_training_img == 1]
        else:
            self.data = self.data[self.data.is_training_img == 0]

        x = os.path.join(self.root, 'CUB_200_2011', 'images') + "/" + np.array(data["filepath"])
        y = np.array(data["target"])  - 1  # Targets start at 1 by default, so shift to 0

        self.dataset = [x, y, None]

    def _check_integrity(self):
        try:
            self._load_metadata()
        except Exception:
            return False

        for index, row in self.data.iterrows():
            filepath = os.path.join(self.root, self.base_folder, row.filepath)
            if not os.path.isfile(filepath):
                print(filepath)
                return False
        return True

    def _download(self):
        import tarfile

        if self._check_integrity():
            print('Files already downloaded and verified')
            return

        download_url(self.url, self.root, self.filename, self.tgz_md5)

        with tarfile.open(os.path.join(self.root, self.filename), "r:gz") as tar:
            tar.extractall(path=self.root)

    def __len__(self):
        return len(self.data)


    def get_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self.dataset