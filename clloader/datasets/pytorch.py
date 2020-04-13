import abc
from typing import List, Tuple, Union

import numpy as np

from clloader.datasets import PyTorchDataset
from torchvision import datasets as torchdata
from torchvision import transforms


class CIFAR10(PyTorchDataset):
    dataset_type = torchdata.cifar.CIFAR10

    @property
    def transformations(self):
        return [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ]


class CIFAR100(PyTorchDataset):
    dataset_type = torchdata.cifar.CIFAR100

    @property
    def transformations(self):
        return [
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        ]


class MNIST(PyTorchDataset):
    dataset_type = torchdata.MNIST


class FashionMNIST(PyTorchDataset):
    dataset_type = torchdata.FashionMNIST
