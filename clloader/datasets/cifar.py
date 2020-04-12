import abc
from typing import List, Tuple, Union

import numpy as np

from clloader.datasets import PyTorchDataset
from torchvision import datasets as torchdata


class CIFAR10(PyTorchDataset):
    dataset_type = torchdata.cifar.CIFAR10


class CIFAR100(PyTorchDataset):
    dataset_type = torchdata.cifar.CIFAR100
