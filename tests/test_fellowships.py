import pytest
import numpy as np

from continuum import ClassIncremental, InstanceIncremental
from continuum.datasets import InMemoryDataset
from continuum.datasets import (
    CIFAR10, CIFAR100, KMNIST, MNIST, CIFARFellowship, FashionMNIST, Fellowship, MNISTFellowship
)


@pytest.mark.slow
def test_MNIST_Fellowship_Instance_Incremental(tmpdir):
    scenario = MNISTFellowship(data_path=tmpdir, train=True, download=True)
    scenario.get_data()
    continuum = InstanceIncremental(scenario, nb_tasks=3)
    assert len(continuum) == 3


@pytest.mark.slow
def test_MNIST_Fellowship_nb_classes(tmpdir):
    scenario = MNISTFellowship(data_path=tmpdir, train=True, download=True)
    x, y, t = scenario.get_data()
    assert len(np.unique(y)) == 30
    scenario = None
    scenario = MNISTFellowship(data_path=tmpdir, train=True, download=True, update_labels=False)
    x, y, t = scenario.get_data()
    assert len(np.unique(y)) == 10


@pytest.mark.slow
def test_MNIST_Fellowship(tmpdir):
    scenario = MNISTFellowship(data_path=tmpdir, train=True, download=True)
    scenario.get_data()
    continuum = ClassIncremental(scenario, increment=10)
    assert len(continuum) == 3


@pytest.mark.slow
def test_CIFAR_Fellowship(tmpdir):
    cl_dataset = CIFARFellowship(data_path=tmpdir, train=True, download=True)
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
def test_Fellowship_classes(tmpdir, list_datasets, nb_tasks):
    cl_dataset = Fellowship(data_path=tmpdir, dataset_list=list_datasets)
    scenario = ClassIncremental(cl_dataset, increment=10)

    assert len(scenario) == nb_tasks
    for task_id, taskset in enumerate(scenario):
        classes = taskset.get_classes()

        # we check if all classes are here
        assert len(classes) == (classes.max() - classes.min() + 1)


@pytest.mark.slow
@pytest.mark.parametrize("list_datasets", [[MNIST, CIFAR10]])
def test_Fellowship_Dimension_Fail(tmpdir, list_datasets):
    cl_dataset = Fellowship(data_path=tmpdir, dataset_list=list_datasets)

    # This does not work since CIFAR10 and MNIST data are not same shape
    with pytest.raises(ValueError):
        continuum = ClassIncremental(cl_dataset, increment=10)
