from continuum.datasets import _ContinuumDataset
from typing import Tuple
import numpy as np
from matplotlib import pyplot as plt

class BackgroundSwap:

    def __init__(self, bg_images: _ContinuumDataset, input_dim: Tuple[int, int] = (28, 28)) -> None:
        """
        :param bg_images: background image set.
        :param input_dim: input dimension of transform, excluding channels
        """
        self.input_dim = input_dim
        self.bg_images = bg_images.get_data()[0]

    def _randcrop(self, img: np.ndarray) -> np.ndarray:
        x_crop = np.random.randint(0, img.shape[0] - self.input_dim[0])
        y_crop = np.random.randint(0, img.shape[1] - self.input_dim[1])

        return img[x_crop:x_crop + self.input_dim[0], y_crop: y_crop + self.input_dim[1], :]

    def __call__(self, img: np.ndarray, mask: bool = None) -> np.ndarray:
        """
        :param img: input image
        :param mask: boolean mask for the foreground of img, if None then a .5 threshold is used
        """

        # TODO: don't assume img is 2 dimensional (h, w)

        img = np.expand_dims(img, 2)
        img = img / 255.0
        img = np.concatenate([img, img, img], axis=2)

        if mask is None:
            mask = (img > .5)

        bg = self.bg_images[np.random.randint(0, len(self.bg_images))] / 255.0
        bg = self._randcrop(bg)

        return mask * img + np.ma.logical_not(mask) * bg
