# pylint: disable=C0401
# flake8: noqa
from continuum.datasets.base import (
    ImageFolderDataset, InMemoryDataset, PyTorchDataset, _ContinuumDataset
)
from continuum.datasets.core50 import CORe50
from continuum.datasets.fellowship import (CIFARFellowship, Fellowship, MNISTFellowship)
from continuum.datasets.imagenet import ImageNet100, ImageNet1000
from continuum.datasets.nlp import MultiNLI
from continuum.datasets.pytorch import (
    CIFAR10, CIFAR100, EMNIST, KMNIST, MNIST, QMNIST, FashionMNIST
)
from continuum.datasets.transformed import PermutedMNIST, RotatedMNIST
