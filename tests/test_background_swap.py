from continuum.transforms import segmentation
from continuum.datasets import CIFAR10
from continuum.datasets import MNIST

def test_background_swap():
    mnist = MNIST("MNIST_DATA", download=True, train=True)
    cifar = CIFAR10("CIFAR10_DATA", download=True, train=True)

    bg_swap = segmentation.BackgroundSwap(cifar, lambda x: x > .5)

    im = mnist.get_data()[0][0]
    im = bg_swap(im)

