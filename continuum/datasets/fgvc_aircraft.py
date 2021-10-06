import os
from typing import Tuple

import numpy as np

from continuum.datasets import _ContinuumDataset
from continuum.download import download, untar
from continuum.tasks import TaskType


class FGVCAircraft(_ContinuumDataset):
    """FGVC-Aircraft 2013 dataset.

    :param target: What kind of target to use, among variants (100 classes),
                   manufacturers (40), or families (70).

    * Fine-Grained Visual Classification of Aircraft},
      S. Maji and J. Kannala and E. Rahtu and M. Blaschko and A. Vedaldi
      arXiv 2013
    """
    url = "https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/archives/fgvc-aircraft-2013b.tar.gz"


    def __init__(self, data_path, train: bool = True, download: bool = True, target: str = "variants"):
        super().__init__(data_path, train, download)

        if target not in ("variants", "manufacturers", "families"):
            raise ValueError(f"Unsupported target <{target}>, available are "
                             " <variants>, <manufacturers>, and <families>.")
        self.target = target

    @property
    def data_type(self) -> TaskType:
        return TaskType.IMAGE_PATH

    def _download(self):
        if not os.path.exists(os.path.join(self.data_path, "fgvc-aircraft-2013b")):
            archive_path = os.path.join(self.data_path, "fgvc-aircraft-2013b.tar.gz")

            if not os.path.exists(archive_path):
                print(f"Downloading archive ...", end=" ")
                download(self.url, self.data_path)
                print('Done!')

            print(f"Extracting archive...", end=" ")
            untar(archive_path)
            print("Done!")

    def get_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.target == "variants":
            a, b = "variants.txt", "variant"
        elif self.target == "manufacturers":
            a, b = "manufacturers.txt", "manufacturer"
        else:
            a, b = "families.txt", "family"

        if self.train:
            c = "trainval"
        else:
            c = "test"

        with open(os.path.join(self.data_path, "fgvc-aircraft-2013b", "data", a)) as f:
            classes = list(map(lambda x: x.strip().replace(" ", ""), f.readlines()))

        x, y = [], []
        with open(os.path.join(self.data_path, "fgvc-aircraft-2013b", "data", f"images_{b}_{c}.txt")) as f:
            for line in f:
                image_id = line[:7].strip()
                x.append(
                    os.path.join(self.data_path, "fgvc-aircraft-2013b", "data", "images", f"{image_id}.jpg")
                )
                y.append(
                    classes.index(line[7:].strip().replace(" ", ""))
                )


        return np.array(x), np.array(y), None
