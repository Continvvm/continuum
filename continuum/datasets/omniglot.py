import os

import torchvision
import numpy as np

from continuum.datasets import _ContinuumDataset
from continuum.tasks import TaskType


class Omniglot(_ContinuumDataset):
    """Omniglot dataset.

    964 characters/classes coming from 30 different alphabets/domains.


    """
    def __init__(
        self,
        data_path: str = "",
        train: bool = True,
        download: bool = True
    ):
        super().__init__(
            data_path,
            train=train,
            download=download,
        )

        if not train:
            warnings.warn("Omniglot has not train/test set, serving train set instead.")

        self.dataset = torchvision.datasets.Omniglot(
            self.data_path, download=self.download, background=self.train
        )

    @property
    def data_type(self) -> TaskType:
        return TaskType.IMAGE_PATH

    def get_data(self):
        x, y, t = [], [], []

        for image_name, char_index in self.dataset._flat_character_images:
            char_name = self.dataset._characters[char_index]

            x.append(
                os.path.join(
                    self.data_path,
                    "omniglot-py/images_background",
                    char_name,
                    image_name
                )
            )
            y.append(char_index)
            t.append(self.dataset._alphabets.index(char_name.split("/")[0]))

        x = np.array(x)
        y = np.array(y)
        t = np.array(t)

        return x, y, t
