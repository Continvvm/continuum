from continuum.transforms.custom import BackgroundSwap
from continuum.datasets import CIFAR10
from continuum.datasets import MNIST
import torchvision
from continuum.scenarios import TransformationIncremental
from matplotlib import pyplot as plt
import pytest


@pytest.mark.slow
def test_background_swap_numpy():
    mnist = MNIST("MNIST_DATA", download=True, train=True)
    cifar = CIFAR10("CIFAR10_DATA", download=True, train=True)

    bg_swap = BackgroundSwap(cifar, input_dim=(28, 28))

    im = mnist.get_data()[0][0]
    im = bg_swap(im)


@pytest.mark.slow
def test_background_swap_torch():
    cifar = CIFAR10("CIFAR10_DATA", download=True, train=True)

    mnist = torchvision.datasets.MNIST('./TorchMnist/', train=True, download=True,
                                       transform=torchvision.transforms.Compose([
                                           torchvision.transforms.ToTensor(),
                                           torchvision.transforms.Normalize(
                                               (0.5,), (0.5,))
                                       ]))

    bg_swap = BackgroundSwap(cifar, input_dim=(28, 28), normalize_input=False)
    im = mnist[0][0]

    im = bg_swap(im)



def test_transform_incremental_bg_swap():
    pass
