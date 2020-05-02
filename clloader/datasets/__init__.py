from clloader.datasets.base import (
    ImageFolderDataset, InMemoryDataset, PyTorchDataset, _ContinuumDataset
)
from clloader.datasets.core50 import CORe50
from clloader.datasets.fellowship import (CIFARFellowship, Fellowship, MNISTFellowship)
from clloader.datasets.imagenet import ImageNet100, ImageNet1000
from clloader.datasets.nlp import MultiNLI
from clloader.datasets.pytorch import (
    CIFAR10, CIFAR100, EMNIST, KMNIST, MNIST, QMNIST, FashionMNIST
)
from clloader.datasets.transformed import PermutedMNIST, RotatedMNIST

# yapf: disable
__all__ = [
    _ContinuumDataset,
    PyTorchDataset,
    InMemoryDataset,
    ImageFolderDataset,
    CIFAR10,
    CIFAR100,
    MNIST,
    FashionMNIST,
    KMNIST,
    EMNIST,
    QMNIST,
    ImageNet100,
    ImageNet1000,
    PermutedMNIST,
    RotatedMNIST,
    Fellowship,
    MNISTFellowship,
    CIFARFellowship,
    CORe50,
    MultiNLI
]
