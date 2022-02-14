from continuum.transforms.custom import BackgroundSwap
from continuum.datasets import CIFAR10, InMemoryDataset
from continuum.datasets import MNIST
import torchvision
from continuum.scenarios import TransformationIncremental
import pytest
import numpy as np

# Uncomment for debugging via image output
# import matplotlib.pyplot as plt



@pytest.mark.slow
def test_background_swap_numpy():
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
def test_tranform_incremental_order():

    x = np.zeros((20, 2, 2, 3), dtype=np.uint8)
    y = np.ones((20,), dtype=np.int32)
    dummy_dataset = InMemoryDataset(x, y)

    call_order = []

    class DummyTransform:
        def __init__(self, idx):
            self.idx = idx

        def __call__(self, x_in):
            call_order.append(self.idx)
            return x_in

    scenario = TransformationIncremental(dummy_dataset,
                                         base_transformations=[DummyTransform(0)],
                                         incremental_transformations=
                                         [[],
                                          [DummyTransform(1)]])

    assert len(scenario) == 2

    for t in scenario[0]:
        pass

    assert (1 not in call_order) and (0 in call_order)

    call_order = []

    for t in scenario[1]:
        call_order.append(-1)

    assert call_order[-1] == -1
    assert call_order[-2] == 1
    assert call_order[-3] == 0


@pytest.mark.slow
def test_transform_incremental_bg_swap():
    cifar = CIFAR10("CIFAR10_DATA", download=True, train=True)
    mnist = MNIST("MNIST_DATA", download=True, train=True)

    scenario = TransformationIncremental(mnist,
                                         base_transformations=[torchvision.transforms.ToTensor()],
                                         incremental_transformations=
                                         [[],
                                          [BackgroundSwap(cifar, input_dim=(28, 28))]])

    for task_id, task_data in enumerate(scenario):
        for t in task_data:
            pass

