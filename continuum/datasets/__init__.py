# pylint: disable=C0401
# flake8: noqa
from continuum.datasets.base import (
    ImageFolderDataset, InMemoryDataset, PyTorchDataset, _ContinuumDataset
)
from continuum.datasets.core50 import (Core50, Core50v2_79, Core50v2_196, Core50v2_391)
from continuum.datasets.fellowship import (CIFARFellowship, Fellowship, MNISTFellowship)
from continuum.datasets.imagenet import ImageNet100, ImageNet1000
from continuum.datasets.nlp import MultiNLI
from continuum.datasets.pytorch import (
    CIFAR10, CIFAR100, EMNIST, KMNIST, MNIST, QMNIST, FashionMNIST
)
