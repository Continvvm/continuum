from typing import Tuple, Union, Callable, Optional
import numpy as np
import torch
from continuum.datasets import _ContinuumDataset
from skimage.transform import resize
# pylint: disable=invalid-unary-operand-type


class BackgroundSwap:
    """Swap input image background with a randomly selected image from bg_images dataset.

    :param bg_images: background image dataset, must be normalized.
    :param input_dim: input dimension of transform, excluding channels.
    :param bg_label: label class from background image set.
    :param normalize_bg: an optional normalization function.
    """

    def __init__(self, bg_images: _ContinuumDataset, input_dim: Tuple[int, int] = (28, 28),
                 bg_label: Optional[int] = None,
                 normalize_bg: Optional[Callable] = lambda x: x / 255.0):
        self.bg_label = bg_label
        self.normalize_bg = normalize_bg
        self.input_dim = input_dim

        full = bg_images.get_data()

        if bg_label is not None:
            self.bg_images = full[0][np.where(full[1] == bg_label)]
        else:
            self.bg_images = full[0]

    def __call__(self, img: Union[np.ndarray, torch.Tensor],
                 mask: Optional[Union[np.ndarray, torch.BoolTensor]] = None
                 ) -> Union[np.ndarray, torch.Tensor]:
        """Call transform on img input and return swapped bg.

        :param img: input image, must be normalized.
        :param mask: boolean mask for the foreground of img, .5 threshold used by default.
        """
        if isinstance(img, torch.Tensor):
            img = img.repeat(3, 1, 1)
            img = img.permute(1, 2, 0)

        elif isinstance(img, np.ndarray):
            img = np.expand_dims(img, 2)
            img = np.concatenate([img, img, img], axis=2)

        else:
            raise NotImplementedError(f"Input type {type(img)} not implemented")

        new_background: np.ndarray = self.bg_images[np.random.randint(0, len(self.bg_images))]

        if self.normalize_bg:
            new_background = self.normalize_bg(new_background)

        new_background = resize(new_background, self.input_dim)

        mask = (img > .5) if mask is None else mask
        out = mask * img + ~mask * new_background

        if isinstance(out, torch.Tensor):
            out = out.permute(2, 0, 1)

        return out
