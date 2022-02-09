from continuum.transforms.custom import BackgroundSwap
from continuum.datasets import CIFAR10
from continuum.datasets import MNIST
from matplotlib import pyplot as plt


def test_background_swap():
    mnist = MNIST("MNIST_DATA", download=True, train=True)
    cifar = CIFAR10("CIFAR10_DATA", download=True, train=True)

    bg_swap = BackgroundSwap(cifar, input_dim=(28, 28))

    im = mnist.get_data()[0][0]
    im = bg_swap(im)
    plt.imshow(im, interpolation='nearest')
    plt.show()


