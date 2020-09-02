import pytest

from continuum import ClassIncremental
from continuum.datasets import (
    CIFAR10, CIFAR100, KMNIST, MNIST, CIFARFellowship, FashionMNIST, Fellowship, MNISTFellowship
)


@pytest.mark.slow
def test_MNIST_Fellowship():
    scenario = MNISTFellowship(data_path="./tests/Datasets", train=True, download=True)
    scenario.get_data()
    continuum = ClassIncremental(scenario, increment=10)
    assert len(continuum) == 3


@pytest.mark.slow
def test_CIFAR_Fellowship():
    cl_dataset = CIFARFellowship(data_path="./tests/Datasets", train=True, download=True)
    scenario = ClassIncremental(cl_dataset, increment=10)
    assert len(scenario) == 11


@pytest.mark.slow
@pytest.mark.parametrize(
    "list_datasets,nb_tasks", [
        ([MNIST, FashionMNIST], 2),
        ([KMNIST, MNIST, FashionMNIST], 3),
        ([CIFAR10, CIFAR100], 11),
    ]
)
#@pytest.mark.parametrize("shared_label_space", [True, False])
def test_Fellowship_classes(list_datasets, nb_tasks):
    cl_dataset = Fellowship(data_path="./tests/Datasets", dataset_list=list_datasets)
    scenario = ClassIncremental(cl_dataset, increment=10)

    assert len(scenario) == nb_tasks
    for task_id, taskset in enumerate(scenario):
        classes = taskset.get_classes()

        # we check if all classes are here
        assert len(classes) == (classes.max() - classes.min() + 1)


@pytest.mark.slow
@pytest.mark.parametrize("list_datasets", [[MNIST, CIFAR10]])
def test_Fellowship_Dimension_Fail(list_datasets):
    cl_dataset = Fellowship(data_path="./tests/Datasets", dataset_list=list_datasets)

    # This does not work since CIFAR10 and MNIST data are not same shape
    with pytest.raises(ValueError):
        continuum = ClassIncremental(cl_dataset, increment=10)
