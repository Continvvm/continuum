"""Much of the credits goes to Cermelli's code:
- https://github.com/fcdl94/MiB/blob/master/dataset/transform.py
"""
import torchvision.transforms.functional as Fv
import torch
import numpy as np


class Compose:
    """Composes several transforms together.
    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, lbl=None):
        if lbl is not None:
            for t in self.transforms:
                img, lbl = t(img, lbl)
            return img, lbl

        for t in self.transforms:
            img = t(img)
        return img

    def __repr__(self):
        format_string = "Segmentation_" + self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


class ToTensor:
    """Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor.
    Converts a PIL Image or numpy.ndarray (H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]
    if the PIL Image belongs to one of the modes (L, LA, P, I, F, RGB, YCbCr, RGBA, CMYK, 1)
    or if the numpy.ndarray has dtype = np.uint8
    In the other cases, tensors are returned without scaling.
    """
    def __call__(self, pic, lbl=None):
        """
        Note that labels will not be normalized to [0, 1].
        Args:
            pic (PIL Image or numpy.ndarray): Image to be converted to tensor.
            lbl (PIL Image or numpy.ndarray): Label to be converted to tensor.
        Returns:
            Tensor: Converted image and label
        """
        if lbl is not None:
            return Fv.to_tensor(pic), torch.from_numpy(np.array(lbl, dtype=np.uint8))
        else:
            return Fv.to_tensor(pic)

    def __repr__(self):
        return "Segmentation_" + self.__class__.__name__ + '()'
