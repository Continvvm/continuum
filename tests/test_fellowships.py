import numpy as np
import pytest
import os

from continuum.datasets import MNIST, CIFAR10, CIFAR100, KMNIST, FashionMNIST
from continuum.datasets import Fellowship, CIFARFellowship, MNISTFellowship

from continuum import ClassIncremental

@pytest.mark.slow
def test_MNIST_Fellowship():
    cl_dataset = MNISTFellowship(data_path="./pytest/Samples/Datasets", train=True, download=True)


@pytest.mark.slow
def test_CIFAR_Fellowship():
    cl_dataset = CIFARFellowship(data_path="./pytest/Samples/Datasets", train=True, download=True)



@pytest.mark.slow
@pytest.mark.parametrize("list_datasets", [[MNIST, FashionMNIST],
                                           [KMNIST, MNIST, FashionMNIST],
                                           [CIFAR10, CIFAR100],
                                           [KMNIST, MNIST, FashionMNIST, CIFAR10, CIFAR100]])
def test_Fellowship(list_datasets):
    cl_dataset = Fellowship(data_path="./pytest/Samples/Datasets", dataset_list=list_datasets)


@pytest.mark.slow
@pytest.mark.parametrize("list_datasets", [[MNIST, FashionMNIST],
                                           [KMNIST, MNIST, FashionMNIST],
                                           [KMNIST, MNIST, FashionMNIST, CIFAR10]])
def test_Fellowship_classes(list_datasets):
    cl_dataset = Fellowship(data_path="./pytest/Samples/Datasets", dataset_list=list_datasets)
    continuum = ClassIncremental(cl_dataset, increment=10)

    for task_id, taskset in enumerate(continuum):
        taskset
