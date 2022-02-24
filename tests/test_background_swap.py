import os

from torch.utils.data import DataLoader
from continuum.datasets import CIFAR10, InMemoryDataset
from continuum.datasets import MNIST
import torchvision
from continuum.scenarios import TransformationIncremental
import pytest
import numpy as np

from continuum.transforms.bg_swap import BackgroundSwap

DATA_PATH = os.environ.get("CONTINUUM_DATA_PATH")

# Uncomment for debugging via image output
# import matplotlib.pyplot as plt


def test_bg_swap_fast():
    """
    Fast test for background swap.
    """
    bg_x = np.ones(shape=[2, 5, 5, 3]) * -1
    bg_y = np.random.rand(2)

    fg = np.random.normal(loc=.5, scale=.1, size=[5, 5])
    bg = InMemoryDataset(bg_x, bg_y)

    bg_swap = BackgroundSwap(bg, input_dim=(5, 5), normalize_bg=None)

    spliced_1_channel = bg_swap(fg)[:, :, 0]

    assert np.array_equal((spliced_1_channel <= -1), (fg <= .5))


@pytest.mark.slow
def test_background_swap_numpy():
    """
    Test background swap on a single ndarray input.
    """
    mnist = MNIST(DATA_PATH, download=True, train=True)
    cifar = CIFAR10(DATA_PATH, download=True, train=True)

    bg_swap = BackgroundSwap(cifar, input_dim=(28, 28))

    im = mnist.get_data()[0][0]
    im = bg_swap(im)

    # Uncomment for debugging
    # plt.imshow(im, interpolation='nearest')
    # plt.show()


@pytest.mark.slow
def test_background_swap_torch():
    """
    Test background swap on a single tensor input.
    """
    cifar = CIFAR10(DATA_PATH, download=True, train=True)

    mnist = torchvision.datasets.MNIST(DATA_PATH, train=True, download=True,
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
def test_background_tranformation():
    """
    Example code using TransformationIncremental to create a setting with 3 tasks.
    """
    cifar = CIFAR10(DATA_PATH, train=True)
    mnist = MNIST(DATA_PATH, download=False, train=True)
    nb_task = 3
    list_trsf = []
    for i in range(nb_task):
        list_trsf.append([torchvision.transforms.ToTensor(), BackgroundSwap(cifar, bg_label=i, input_dim=(28, 28)),
                          torchvision.transforms.ToPILImage()])
    scenario = TransformationIncremental(mnist, base_transformations=[torchvision.transforms.ToTensor()],
                                         incremental_transformations=list_trsf)
    folder = "tests/samples/background_trsf/"
    if not os.path.exists(folder):
        os.makedirs(folder)
    for task_id, task_data in enumerate(scenario):
        task_data.plot(path=folder, title=f"background_{task_id}.jpg", nb_samples=100, shape=[28, 28, 3])
        loader = DataLoader(task_data)
        _, _, _ = next(iter(loader))
