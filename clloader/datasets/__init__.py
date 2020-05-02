from clloader.datasets.base import (
    ImageFolderDataset, InMemoryDataset, PyTorchDataset, _ContinuumDataset
)
from clloader.datasets.fellowship import (CIFARFellowship, Fellowship, MNISTFellowship)
from clloader.datasets.imagenet import ImageNet100, ImageNet1000
from clloader.datasets.pytorch import (
    CIFAR10, CIFAR100, EMNIST, KMNIST, MNIST, QMNIST, FashionMNIST
)
from clloader.datasets.transformed import PermutedMNIST, RotatedMNIST
