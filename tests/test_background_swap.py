from continuum.transforms.custom import BackgroundSwap
from continuum.datasets import CIFAR10, InMemoryDataset
from continuum.datasets import MNIST
import torchvision
from continuum.scenarios import TransformationIncremental
import pytest


# Uncomment for debugging via image output
# import matplotlib.pyplot as plt


@pytest.mark.slow
def test_background_swap_numpy():
    """
    Test background swap on a single ndarray input
    """
    mnist = MNIST("MNIST_DATA", download=True, train=True)
    cifar = CIFAR10("CIFAR10_DATA", download=True, train=True)

    bg_swap = BackgroundSwap(cifar, input_dim=(28, 28))

    im = mnist.get_data()[0][0]
    im = bg_swap(im)

    # Uncomment for debugging
    # plt.imshow(im, interpolation='nearest')
    # plt.show()


@pytest.mark.slow
def test_background_swap_torch():
    """
    Test background swap on a single tensor input
    """
    cifar = CIFAR10("CIFAR10_DATA", download=True, train=True)

    mnist = torchvision.datasets.MNIST('./TorchMnist/', train=True, download=True,
                                       transform=torchvision.transforms.Compose([
                                           torchvision.transforms.ToTensor()
                                       ]))

    bg_swap = BackgroundSwap(cifar, input_dim=(28, 28))
    im = mnist[0][0]

    im = bg_swap(im)

    # Uncomment for debugging
    # plt.imshow(im.permute(1, 2, 0), interpolation='nearest')
    # plt.show()


@pytest.mark.slow
def test_transform_incremental_bg_swap():
    """
    Test Background swap transform on a full mnist dataset with cifar as background
    """
    cifar = CIFAR10("CIFAR10_DATA", download=True, train=True)
    mnist = MNIST("MNIST_DATA", download=True, train=True)

    scenario = TransformationIncremental(mnist,
                                         base_transformations=None,
                                         incremental_transformations=[[torchvision.transforms.ToTensor()],
                                                                      [BackgroundSwap(cifar, input_dim=(28, 28))]])

    for task_id, task_data in enumerate(scenario):
        for t in task_data:
            pass
