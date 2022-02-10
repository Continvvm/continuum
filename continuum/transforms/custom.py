from typing import Tuple, Union
import numpy as np
import torch
from PIL.Image import Image
from torch.utils.data import DataLoader
from continuum.datasets import _ContinuumDataset


class BackgroundSwap:
    """
    Swap input image background with a randomly selected image from bg_images dataset

    :param bg_images: background image dataset, must be normalized.
    :param input_dim: input dimension of transform, excluding channels
    :param crop_bg: crop background images to correct size, if false it's assumed they are cropped
    """

    def __init__(self, bg_images: _ContinuumDataset, input_dim: Tuple[int, int] = (28, 28),
                 normalize_bg: bool = True,
                 crop_bg: bool = True):
        self.normalize_bg = normalize_bg
        self.crop_bg = crop_bg
        self.input_dim = input_dim
        self.bg_images = bg_images.get_data()[0]

    def _randcrop(self, img: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """
        Crop input image to self.input_dim shape
        :param img: input image
        """

        crop_height = img.shape[0] >= self.input_dim[0]
        crop_width = img.shape[1] >= self.input_dim[1]

        x_crop = np.random.randint(0, img.shape[0] - self.input_dim[0])
        y_crop = np.random.randint(0, img.shape[1] - self.input_dim[1])

        if crop_width and crop_height:
            return img[x_crop:x_crop + self.input_dim[0], y_crop: y_crop + self.input_dim[1], :]

        elif crop_width:
            return img[:, y_crop: y_crop + self.input_dim[1], :]

        elif crop_height:
            return img[x_crop:x_crop + self.input_dim[0], :, :]

        else:
            raise Exception("Background image is smaller than foreground image")

    def __call__(self, img: Union[np.ndarray, torch.Tensor],
                 mask: Union[np.ndarray, torch.BoolTensor] = None) -> Union[np.ndarray, torch.Tensor]:
        """
        Splice input image foreground with randomly sampled background.

        Inputting a torch.Tensor assumes the channel dim comes first,
        while inputting a np.ndarray requires the channel dim to come second

        :param img: input image, must be normalized
        :param mask: boolean mask for the foreground of img, if None then a .5 threshold is used
        """

        if isinstance(img, torch.Tensor):
            img = img.repeat(3, 1, 1)
            img = img.permute(1, 2, 0)

        elif isinstance(img, np.ndarray):
            img = np.expand_dims(img, 2)
            img = np.concatenate([img, img, img], axis=2)

        else:
            raise NotImplementedError("Input type not implemented")

        new_background = self.bg_images[np.random.randint(0, len(self.bg_images))]

        if self.normalize_bg:
            # TODO: don't hardcode normalization
            new_background = new_background / 255.0

        if self.crop_bg:
            new_background = self._randcrop(new_background)

        if mask is None:
            mask = (img > .5)

        out = mask * img + ~mask * new_background

        if isinstance(out, torch.Tensor):
            out = out.permute(2, 0, 1)

        return out


