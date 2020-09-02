import numpy as np
import pytest

from continuum.scenarios import ClassIncremental
from continuum.datasets import MNIST, CIFAR10, CIFAR100, KMNIST, FashionMNIST
from torchvision.transforms import transforms


# yapf: disable


@pytest.mark.slow
@pytest.mark.parametrize("dataset, increment", [(MNIST, 5),
                                                (KMNIST, 2),
                                                (FashionMNIST, 1),
                                                (CIFAR10, 2),
                                                (CIFAR100, 10)])
def test_with_dataset_simple_increment(dataset, increment):
    dataset = dataset(data_path="./tests/Datasets", download=True, train=True)
    scenario = ClassIncremental(cl_dataset=dataset,
                                 increment=increment,
                                 transformations=[transforms.ToTensor()]
                                 )

    for task_id, taskset in enumerate(scenario):
        classes = taskset.get_classes()

        assert len(classes) == increment

        # check if there is continuity in classes by default
        assert len(classes) == (classes.max() - classes.min() + 1)


@pytest.mark.slow
@pytest.mark.parametrize("dataset, increment", [(MNIST, [5, 1, 1, 3]),
                                                (KMNIST, [2, 2, 4, 2]),
                                                (FashionMNIST, [1, 2, 1, 2, 1, 2, 1]),
                                                (CIFAR10, [2, 2, 2, 2, 2]),
                                                (CIFAR100, [50, 10, 20, 20])])
def test_with_dataset_composed_increment(dataset, increment):
    dataset = dataset(data_path="./tests/Datasets", download=True, train=True)
    scenario = ClassIncremental(cl_dataset=dataset,
                                 increment=increment,
                                 transformations=[transforms.ToTensor()]
                                 )

    for task_id, taskset in enumerate(scenario):
        classes = taskset.get_classes()

        assert len(classes) == increment[task_id]

        # check if there is continuity in classes by default
        assert len(classes) == (classes.max() - classes.min() + 1)
