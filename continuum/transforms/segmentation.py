"""This module is mostly a copy/paste from the useful MiB codebase of Cermelli et al.
See there:
- https://github.com/fcdl94/MiB/blob/master/dataset/transform.py
"""

import warnings
import math
import random
import numbers
import collections

import torch
import torchvision.transforms.functional as Fv
import numpy as np
from PIL import Image


_pil_interpolation_to_str = {
    Image.NEAREST: 'PIL.Image.NEAREST',
    Image.BILINEAR: 'PIL.Image.BILINEAR',
    Image.BICUBIC: 'PIL.Image.BICUBIC',
    Image.LANCZOS: 'PIL.Image.LANCZOS',
    Image.HAMMING: 'PIL.Image.HAMMING',
    Image.BOX: 'PIL.Image.BOX',
}


class Compose:
    """Composes several transforms together.

    :param transforms: list of transforms to compose.
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, lbl=None):
        if lbl is not None:
            for t in self.transforms:
                img, lbl = t(img, lbl)
            return img, lbl
        else:
            for t in self.transforms:
                img = t(img)
            return img

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


class Resize:
    """Resize the input PIL Image to the given size.

    :param size: (sequence or int): Desired output size. If size is a sequence like
                 (h, w), output size will be matched to this. If size is an int,
                 smaller edge of the image will be matched to this number.
                 i.e, if height > width, then image will be rescaled to
                 (size * height / width, size)
    :param interpolation: (int, optional): Desired interpolation. Default is
                          ``PIL.Image.BILINEAR``
    """

    def __init__(self, size, interpolation=Image.BILINEAR):
        assert isinstance(size, int) or (isinstance(size, collections.Iterable) and len(size) == 2)
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img, lbl=None):
        """
        :param img: (PIL Image): Image to be scaled.
        :return: PIL Image: Rescaled image.
        """
        if lbl is not None:
            return Fv.resize(img, self.size, self.interpolation), Fv.resize(lbl, self.size, Image.NEAREST)
        else:
            return Fv.resize(img, self.size, self.interpolation)

    def __repr__(self):
        interpolate_str = _pil_interpolation_to_str[self.interpolation]
        return self.__class__.__name__ + '(size={0}, interpolation={1})'.format(self.size, interpolate_str)


class CenterCrop:
    """Crops the given PIL Image at the center.

    :param size: (sequence or int): Desired output size of the crop. If size is an
                 int instead of sequence like (h, w), a square crop (size, size) is
                 made.
    """

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img, lbl=None):
        """
        :param img: (PIL Image): Image to be cropped.
        :return: PIL Image: Cropped image.
        """
        if lbl is not None:
            return Fv.center_crop(img, self.size), Fv.center_crop(lbl, self.size)
        else:
            return Fv.center_crop(img, self.size)

    def __repr__(self):
        return self.__class__.__name__ + f'(size={self.size})'


class Pad:
    """Pad the given PIL Image on all sides with the given "pad" value.

    :param padding: (int or tuple): Padding on each border. If a single int is provided this
                    is used to pad all borders. If tuple of length 2 is provided this is the padding
                    on left/right and top/bottom respectively. If a tuple of length 4 is provided
                    this is the padding for the left, top, right and bottom borders
                    respectively.
    :param fill: (int): Pixel fill value for constant fill. Default is 0.
                 This value is only used when the padding_mode is constant
    :param padding_mode: (str): Type of padding. Should be: constant, edge, reflect or symmetric.
                         Default is constant.
                        - constant: pads with a constant value, this value is specified with fill
                        - edge: pads with the last value at the edge of the image
                        - reflect: pads with reflection of image without repeating the last value on the edge
                            Fvor example, padding [1, 2, 3, 4] with 2 elements on both sides in reflect mode
                            will result in [3, 2, 1, 2, 3, 4, 3, 2]
                        - symmetric: pads with reflection of image repeating the last value on the edge
                            Fvor example, padding [1, 2, 3, 4] with 2 elements on both sides in symmetric mode
                            will result in [2, 1, 1, 2, 3, 4, 4, 3]
    """

    def __init__(self, padding, fill=0, padding_mode='constant'):
        assert isinstance(padding, (numbers.Number, tuple))
        assert isinstance(fill, (numbers.Number, str))
        assert padding_mode in ['constant', 'edge', 'reflect', 'symmetric']
        if isinstance(padding, collections.Sequence) and len(padding) not in [2, 4]:
            raise ValueError("Padding must be an int or a 2, or 4 element tuple, not a " +
                             "{} element tuple".format(len(padding)))

        self.padding = padding
        self.fill = fill
        self.padding_mode = padding_mode

    def __call__(self, img, lbl=None):
        """
        :param img: (PIL Image): Image to be padded.
        :return: PIL Image: Padded image.
        """
        if lbl is not None:
            return Fv.pad(img, self.padding, self.fill, self.padding_mode), Fv.pad(lbl, self.padding, self.fill, self.padding_mode)
        else:
            return Fv.pad(img, self.padding, self.fill, self.padding_mode)

    def __repr__(self):
        return self.__class__.__name__ + '(padding={0}, fill={1}, padding_mode={2})'.\
            format(self.padding, self.fill, self.padding_mode)


class Lambda:
    """Apply a user-defined lambda as a transform.

    :param lambd: (function): Lambda/function to be used for transform.
    """

    def __init__(self, lambd):
        assert callable(lambd), repr(type(lambd).__name__) + " object is not callable"
        self.lambd = lambd

    def __call__(self, img, lbl=None):
        if lbl is not None:
            return self.lambd(img), self.lambd(lbl)
        else:
            return self.lambd(img)

    def __repr__(self):
        return self.__class__.__name__ + '()'


class RandomRotation:
    """Rotate the image by angle.

    :param degrees: (sequence or float or int): Range of degrees to select from.
                    If degrees is a number instead of sequence like (min, max), the range of degrees
                    will be (-degrees, +degrees).
    :param resample: ({PIL.Image.NEAREST, PIL.Image.BILINEAR, PIL.Image.BICUBIC}, optional):
                     An optional resampling filter.
                     See http://pillow.readthedocs.io/en/3.4.x/handbook/concepts.html#filters
                     If omitted, or if the image has mode "1" or "P", it is set to PIL.Image.NEAREST.
    :param expand: (bool, optional): Optional expansion flag.
                   If true, expands the output to make it large enough to hold the entire rotated image.
                   If false or omitted, make the output image the same size as the input image.
                   Note that the expand flag assumes rotation around the center and no translation.
    :param center: (2-tuple, optional): Optional center of rotation.
                   Origin is the upper left corner.
                   Default is the center of the image.
    """

    def __init__(self, degrees, resample=False, expand=False, center=None):
        if isinstance(degrees, numbers.Number):
            if degrees < 0:
                raise ValueError("If degrees is a single number, it must be positive.")
            self.degrees = (-degrees, degrees)
        else:
            if len(degrees) != 2:
                raise ValueError("If degrees is a sequence, it must be of len 2.")
            self.degrees = degrees

        self.resample = resample
        self.expand = expand
        self.center = center

    @staticmethod
    def get_params(degrees):
        """Get parameters for ``rotate`` for a random rotation.

        :return: sequence: params to be passed to ``rotate`` for random rotation.
        """
        angle = random.uniform(degrees[0], degrees[1])

        return angle

    def __call__(self, img, lbl):
        """

        :param img: (PIL Image): Image to be rotated.
        :param lbl: (PIL Image): Label to be rotated.
        :return: PIL Image: Rotated image, PIL Image: Rotated label.
        """

        angle = self.get_params(self.degrees)
        if lbl is not None:
            return Fv.rotate(img, angle, self.resample, self.expand, self.center), \
                   Fv.rotate(lbl, angle, self.resample, self.expand, self.center)
        else:
            return Fv.rotate(img, angle, self.resample, self.expand, self.center)

    def __repr__(self):
        format_string = self.__class__.__name__ + '(degrees={0}'.format(self.degrees)
        format_string += ', resample={0}'.format(self.resample)
        format_string += ', expand={0}'.format(self.expand)
        if self.center is not None:
            format_string += ', center={0}'.format(self.center)
        format_string += ')'
        return format_string


class RandomHorizontalFvlip:
    """Horizontally flip the given PIL Image randomly with a given probability.

    :param p: (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, lbl=None):
        """
        :param img: (PIL Image): Image to be flipped.
        :return: PIL Image: Randomly flipped image.
        """
        if random.random() < self.p:
            if lbl is not None:
                return Fv.hflip(img), Fv.hflip(lbl)
            else:
                return Fv.hflip(img)
        if lbl is not None:
            return img, lbl
        else:
            return img

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


class RandomVerticalFvlip:
    """Vertically flip the given PIL Image randomly with a given probability.

    :param p: (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, lbl):
        """
        :param img: (PIL Image): Image to be flipped.
        :param lbl: (PIL Image): Label to be flipped.
        :return: PIL Image: Randomly flipped image, PIL Image: Randomly flipped label.
        """
        if random.random() < self.p:
            if lbl is not None:
                return Fv.vflip(img), Fv.vflip(lbl)
            else:
                return Fv.vflip(img)
        if lbl is not None:
            return img, lbl
        else:
            return img

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


class ToTensor:
    """Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor.

    Converts a PIL Image or numpy.ndarray (H x W x C) in the range
    [0, 255] to a torch.FvloatTensor of shape (C x H x W) in the range [0.0, 1.0]
    if the PIL Image belongs to one of the modes (L, LA, P, I, Fv, RGB, YCbCr, RGBA, CMYK, 1)
    or if the numpy.ndarray has dtype = np.uint8
    In the other cases, tensors are returned without scaling.

    """

    def __call__(self, pic, lbl=None):
        """
        Note that labels will not be normalized to [0, 1].

        :param pic: (PIL Image or numpy.ndarray): Image to be converted to tensor.
        :param lbl: (PIL Image or numpy.ndarray): Label to be converted to tensor.
        :return: Tensor: Converted image and label
        """
        if lbl is not None:
            return Fv.to_tensor(pic), torch.from_numpy(np.array(lbl, dtype=np.uint8))
        else:
            return Fv.to_tensor(pic)

    def __repr__(self):
        return self.__class__.__name__ + '()'


class Normalize:
    """Normalize a tensor image with mean and standard deviation.
    Given mean: ``(M1,...,Mn)`` and std: ``(S1,..,Sn)`` for ``n`` channels, this transform
    will normalize each channel of the input ``torch.*Tensor`` i.e.
    ``input[channel] = (input[channel] - mean[channel]) / std[channel]``

    :param mean: (sequence): Sequence of means for each channel.
    :param std: (sequence): Sequence of standard deviations for each channel.
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor, lbl=None):
        """
        :param tensor: (Tensor): Tensor image of size (C, H, W) to be normalized.
        :param lbl: (Tensor): Tensor of label. A dummy input for ExtCompose
        :return: Tensor: Normalized Tensor image, Tensor: Unchanged Tensor label
        """
        if lbl is not None:
            return Fv.normalize(tensor, self.mean, self.std), lbl
        else:
            return Fv.normalize(tensor, self.mean, self.std)

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


class RandomCrop:
    """Crop the given PIL Image at a random location.

    :param size: (sequence or int): Desired output size of the crop. If size is an
                 int instead of sequence like (h, w), a square crop (size, size) is
                 made.
    :param padding: (int or sequence, optional): Optional padding on each border
                    of the image. Default is 0, i.e no padding. If a sequence of length
                    4 is provided, it is used to pad left, top, right, bottom borders
                    respectively.
    :param pad_if_needed: (boolean): It will pad the image if smaller than the
                          desired size to avoid raising an exception.
    """

    def __init__(self, size, padding=0, pad_if_needed=False):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding
        self.pad_if_needed = pad_if_needed

    @staticmethod
    def get_params(img, output_size):
        """Get parameters for ``crop`` for a random crop.

        :param img: (PIL Image): Image to be cropped.
        :param output_size: (tuple): Expected output size of the crop.
        :return: tuple: params (i, j, h, w) to be passed to ``crop`` for random crop.
        """
        w, h = img.size
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w

        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return i, j, th, tw

    def __call__(self, img, lbl=None):
        """
        :param img: (PIL Image): Image to be cropped.
        :param lbl: (PIL Image): Label to be cropped.
        :return: PIL Image: Cropped image, PIL Image: Cropped label.
        """
        if lbl is None:
            if self.padding > 0:
                img = Fv.pad(img, self.padding)
            # pad the width if needed
            if self.pad_if_needed and img.size[0] < self.size[1]:
                img = Fv.pad(img, padding=int((1 + self.size[1] - img.size[0]) / 2))
            # pad the height if needed
            if self.pad_if_needed and img.size[1] < self.size[0]:
                img = Fv.pad(img, padding=int((1 + self.size[0] - img.size[1]) / 2))

            i, j, h, w = self.get_params(img, self.size)

            return Fv.crop(img, i, j, h, w)

        else:
            assert img.size == lbl.size, 'size of img and lbl should be the same. %s, %s' % (img.size, lbl.size)
            if self.padding > 0:
                img = Fv.pad(img, self.padding)
                lbl = Fv.pad(lbl, self.padding)

            # pad the width if needed
            if self.pad_if_needed and img.size[0] < self.size[1]:
                img = Fv.pad(img, padding=int((1 + self.size[1] - img.size[0]) / 2))
                lbl = Fv.pad(lbl, padding=int((1 + self.size[1] - lbl.size[0]) / 2))

            # pad the height if needed
            if self.pad_if_needed and img.size[1] < self.size[0]:
                img = Fv.pad(img, padding=int((1 + self.size[0] - img.size[1]) / 2))
                lbl = Fv.pad(lbl, padding=int((1 + self.size[0] - lbl.size[1]) / 2))

            i, j, h, w = self.get_params(img, self.size)

            return Fv.crop(img, i, j, h, w), Fv.crop(lbl, i, j, h, w)

    def __repr__(self):
        return self.__class__.__name__ + '(size={0}, padding={1})'.format(self.size, self.padding)


class RandomResizedCrop:
    """Crop the given PIL Image to random size and aspect ratio.
    A crop of random size (default: of 0.08 to 1.0) of the original size and a random
    aspect ratio (default: of 3/4 to 4/3) of the original aspect ratio is made. This crop
    is finally resized to given size.
    This is popularly used to train the Inception networks.

    :param size: expected output size of each edge
    :param scale: range of size of the origin size cropped
    :param ratio: range of aspect ratio of the origin aspect ratio cropped
    :param interpolation: Default: PIL.Image.BILINEAR
    """

    def __init__(self, size, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.), interpolation=Image.BILINEAR):
        if isinstance(size, tuple):
            self.size = size
        else:
            self.size = (size, size)
        if (scale[0] > scale[1]) or (ratio[0] > ratio[1]):
            warnings.warn("range should be of kind (min, max)")

        self.interpolation = interpolation
        self.scale = scale
        self.ratio = ratio

    @staticmethod
    def get_params(img, scale, ratio):
        """Get parameters for ``crop`` for a random sized crop.

        :param img: (PIL Image): Image to be cropped.
        :param scale: (tuple): range of size of the origin size cropped
        :param ratio: (tuple): range of aspect ratio of the origin aspect ratio cropped
        :return tuple: params (i, j, h, w) to be passed to ``crop`` for a random
                       sized crop.
        """
        area = img.size[0] * img.size[1]

        for attempt in range(10):
            target_area = random.uniform(*scale) * area
            log_ratio = (math.log(ratio[0]), math.log(ratio[1]))
            aspect_ratio = math.exp(random.uniform(*log_ratio))

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if w <= img.size[0] and h <= img.size[1]:
                i = random.randint(0, img.size[1] - h)
                j = random.randint(0, img.size[0] - w)
                return i, j, h, w

        # Fvallback to central crop
        in_ratio = img.size[0] / img.size[1]
        if (in_ratio < min(ratio)):
            w = img.size[0]
            h = int(round(w / min(ratio)))
        elif (in_ratio > max(ratio)):
            h = img.size[1]
            w = int(round(h * max(ratio)))
        else:  # whole image
            w = img.size[0]
            h = img.size[1]
        i = (img.size[1] - h) // 2
        j = (img.size[0] - w) // 2
        return i, j, h, w

    def __call__(self, img, lbl=None):
        """
        :param img: (PIL Image): Image to be cropped and resized.
        :return: PIL Image: Randomly cropped and resized image.
        """
        i, j, h, w = self.get_params(img, self.scale, self.ratio)
        if lbl is not None:
            return Fv.resized_crop(img, i, j, h, w, self.size, self.interpolation), \
                   Fv.resized_crop(lbl, i, j, h, w, self.size, Image.NEAREST)
        else:
            return Fv.resized_crop(img, i, j, h, w, self.size, self.interpolation)

    def __repr__(self):
        interpolate_str = _pil_interpolation_to_str[self.interpolation]
        format_string = self.__class__.__name__ + '(size={0}'.format(self.size)
        format_string += ', scale={0}'.format(tuple(round(s, 4) for s in self.scale))
        format_string += ', ratio={0}'.format(tuple(round(r, 4) for r in self.ratio))
        format_string += ', interpolation={0})'.format(interpolate_str)
        return format_string


class ColorJitter:
    """Randomly change the brightness, contrast and saturation of an image.

    :param brightness: (float or tuple of float (min, max)): How much to jitter brightness.
                       brightness_factor is chosen uniformly from [max(0, 1 - brightness), 1 + brightness]
                       or the given [min, max]. Should be non negative numbers.
    :param contrast: (float or tuple of float (min, max)): How much to jitter contrast.
                     contrast_factor is chosen uniformly from [max(0, 1 - contrast), 1 + contrast]
                     or the given [min, max]. Should be non negative numbers.
    :param saturation: (float or tuple of float (min, max)): How much to jitter saturation.
                       saturation_factor is chosen uniformly from [max(0, 1 - saturation), 1 + saturation]
                       or the given [min, max]. Should be non negative numbers.
    :param hue: (float or tuple of float (min, max)): How much to jitter hue.
                hue_factor is chosen uniformly from [-hue, hue] or the given [min, max].
                Should have 0<= hue <= 0.5 or -0.5 <= min <= max <= 0.5.
    """
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        self.brightness = self._check_input(brightness, 'brightness')
        self.contrast = self._check_input(contrast, 'contrast')
        self.saturation = self._check_input(saturation, 'saturation')
        self.hue = self._check_input(hue, 'hue', center=0, bound=(-0.5, 0.5),
                                     clip_first_on_zero=False)

    def _check_input(self, value, name, center=1, bound=(0, float('inf')), clip_first_on_zero=True):
        if isinstance(value, numbers.Number):
            if value < 0:
                raise ValueError("If {} is a single number, it must be non negative.".format(name))
            value = [center - value, center + value]
            if clip_first_on_zero:
                value[0] = max(value[0], 0)
        elif isinstance(value, (tuple, list)) and len(value) == 2:
            if not bound[0] <= value[0] <= value[1] <= bound[1]:
                raise ValueError("{} values should be between {}".format(name, bound))
        else:
            raise TypeError("{} should be a single number or a list/tuple with lenght 2.".format(name))

        # if value is 0 or (1., 1.) for brightness/contrast/saturation
        # or (0., 0.) for hue, do nothing
        if value[0] == value[1] == center:
            value = None
        return value

    @staticmethod
    def get_params(brightness, contrast, saturation, hue):
        """Get a randomized transform to be applied on image.
        Arguments are same as that of __init__.

        :return: Transform which randomly adjusts brightness, contrast and
                 saturation in a random order.
        """
        transforms = []

        if brightness is not None:
            brightness_factor = random.uniform(brightness[0], brightness[1])
            transforms.append(Lambda(lambda img: Fv.adjust_brightness(img, brightness_factor)))

        if contrast is not None:
            contrast_factor = random.uniform(contrast[0], contrast[1])
            transforms.append(Lambda(lambda img: Fv.adjust_contrast(img, contrast_factor)))

        if saturation is not None:
            saturation_factor = random.uniform(saturation[0], saturation[1])
            transforms.append(Lambda(lambda img: Fv.adjust_saturation(img, saturation_factor)))

        if hue is not None:
            hue_factor = random.uniform(hue[0], hue[1])
            transforms.append(Lambda(lambda img: Fv.adjust_hue(img, hue_factor)))

        random.shuffle(transforms)
        transform = Compose(transforms)

        return transform

    def __call__(self, img, lbl=None):
        """
        :param img: (PIL Image): Input image.
        :return: PIL Image: Color jittered image.
        """
        transform = self.get_params(self.brightness, self.contrast,
                                    self.saturation, self.hue)
        if lbl is not None:
            return transform(img), lbl
        else:
            return transform(img)

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += 'brightness={0}'.format(self.brightness)
        format_string += ', contrast={0}'.format(self.contrast)
        format_string += ', saturation={0}'.format(self.saturation)
        format_string += ', hue={0})'.format(self.hue)
        return format_string
