from continuum.transforms import segmentation
from continuum.datasets import CIFAR10


def test_background_swap():
    bg_swap = segmentation.BackgroundSwap(CIFAR10)
