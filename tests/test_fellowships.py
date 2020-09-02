import numpy as np
import pytest
import os

from continuum.datasets import MNIST, CIFAR10, CIFAR100, KMNIST, FashionMNIST
from continuum.datasets import Fellowship, CIFARFellowship, MNISTFellowship

from continuum import ClassIncremental

@pytest.mark.slow
def test_MNIST_Fellowship():
    scenario = MNISTFellowship(data_path="./tests/Datasets", train=True, download=True)


@pytest.mark.slow
def test_CIFAR_Fellowship():
    scenario = CIFARFellowship(data_path="./tests/Datasets", train=True, download=True)



@pytest.mark.slow
@pytest.mark.parametrize("list_datasets", [[MNIST, FashionMNIST],
                                           [KMNIST, MNIST, FashionMNIST],
                                           [CIFAR10, CIFAR100],
                                           [KMNIST, MNIST, FashionMNIST, CIFAR10, CIFAR100]])
def test_Fellowship(list_datasets):
    scenario = Fellowship(data_path="./tests/Datasets", dataset_list=list_datasets)


@pytest.mark.slow
@pytest.mark.parametrize("list_datasets", [[MNIST, FashionMNIST],
                                           [KMNIST, MNIST, FashionMNIST]])
#@pytest.mark.parametrize("shared_label_space", [True, False])
def test_Fellowship_classes(list_datasets):
    cl_dataset = Fellowship(data_path="./tests/Datasets", dataset_list=list_datasets)
    scenario = ClassIncremental(cl_dataset, increment=10)

    for task_id, taskset in enumerate(scenario):

        classes = taskset.get_classes()

        # we check if all classes are here
        assert len(classes) == (classes.max()-classes.min()+1)

@pytest.mark.slow
@pytest.mark.parametrize("list_datasets", [[MNIST, CIFAR10]])
def test_Fellowship_Dimension_Fail(list_datasets):
    cl_dataset = Fellowship(data_path="./tests/Datasets", dataset_list=list_datasets)

    # This does not work since CIFAR10 and MNIST data are not same shape
    with pytest.raises(ValueError):
        continuum = ClassIncremental(cl_dataset, increment=10)
