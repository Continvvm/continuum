from continuum.transforms.segmentation import RandomCrop
import numpy as np


class BackgroundSwap:

    def __init__(self, bg_images, input_dim=(28, 28), fg_criterion=None):
        """
        :param bg_images: background image set.
        :param fg_criterion: a boolean function that decides the foreground of the input image.
        """
        self.rand_crop = RandomCrop(size=input_dim)
        self.bg_images = bg_images.get_data()[0]
        self.fg_criterion = fg_criterion

    def __call__(self, img, mask=None):
        """
        :param img: input image
        :param mask: boolean mask for the foreground of img

        NOTE: a mask passed in manually takes precedent over criterion
        """

        if mask is None and self.fg_criterion is None:
            raise Exception('No foreground mask or masking criterion')
        elif mask is None:
            mask = img[self.fg_criterion(img)]  # pseudocode, convert to numpy first

        # TODO: sample from the background set (randomly with replacement?)
        bg = np.random.choice(self.bg_images)

        # TODO: apply random augmentations and crop sampled bg image to dimension of img
        bg = self.rand_crop(bg)

        # TODO: splice img fg with sampled bg image and return
        pass