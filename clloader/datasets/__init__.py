from clloader.datasets.base import (BaseDataset, ImageFolderDataset,
                                    InMemoryDataset, PyTorchDataset)
from clloader.datasets.fellowship import (CIFARFellowship, Fellowship,
                                          MNISTFellowship)
from clloader.datasets.imagenet import ImageNet100, ImageNet1000
from clloader.datasets.pytorch import (CIFAR10, CIFAR100, EMNIST, KMNIST,
                                       MNIST, QMNIST, FashionMNIST)
from clloader.datasets.transformed import PermutedMNIST, RotatedMNIST
from clloader.datasets.continuumDataset import ContinuumDataset, split_train_val

# yapf: disable
__all__ = [
    BaseDataset,
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
    CIFARFellowship
]
