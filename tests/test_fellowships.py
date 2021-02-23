import pytest
import numpy as np
from torch.utils.data import DataLoader

from continuum import ClassIncremental, InstanceIncremental
from continuum.datasets import (
    CIFAR10, CIFAR100, KMNIST, MNIST, CIFARFellowship, FashionMNIST, Fellowship, MNISTFellowship,
    InMemoryDataset, Fellowship
)


@pytest.fixture
def dataset7c():
    return InMemoryDataset(*gen_dataset(7, 0))


@pytest.fixture
def dataset10c():
    return InMemoryDataset(*gen_dataset(10, 1))


@pytest.fixture
def dataset20c():
    return InMemoryDataset(*gen_dataset(20, 2))


def gen_dataset(nb_classes, pixel_value):
    nb_items_per_class = 5

    x_train = np.ones((nb_items_per_class * nb_classes, 32, 32, 3)) * pixel_value
    y_train = []
    for i in range(nb_classes):
        y_train.append(np.ones(nb_items_per_class, dtype=np.int64) * i)
    y_train = np.concatenate(y_train)

    return (x_train, y_train)


@pytest.mark.parametrize("increment", [1, [7, 10, 20]])
def test_inMemory_updateLabels_Fellowship(increment, dataset7c, dataset10c, dataset20c):
    fellow = Fellowship([dataset7c, dataset10c, dataset20c], update_labels=True)

    x, y, t = fellow.get_data()
    assert len(np.unique(t)) == 3
    assert len(np.unique(y)) == 37

    if isinstance(increment, list):
        continuum = ClassIncremental(fellow, increment=increment)
        assert continuum.nb_classes == 37
        assert continuum.nb_tasks == len(increment)
    else:
        continuum = ClassIncremental(fellow, increment=increment)
        assert continuum.nb_tasks == 37
        assert continuum.nb_classes == 37



@pytest.mark.parametrize("increment", [1, [7, 10, 20]])
def test_inMemory_keepLabels_Fellowship(increment, dataset7c, dataset10c, dataset20c):
    fellow = Fellowship([dataset7c, dataset10c, dataset20c], update_labels=False)

    x, y, t = fellow.get_data()
    assert len(np.unique(t)) == 3
    assert len(np.unique(y)) == 20

    if isinstance(increment, list):
        with pytest.raises(Exception):
            continuum = ClassIncremental(fellow, increment=increment)
    else:
        continuum = ClassIncremental(fellow, increment=increment)
        assert continuum.nb_classes == 20
        assert continuum.nb_tasks == 20


@pytest.mark.parametrize("update_labels,nb_tasks", [
    (True, 0),
    (True, 3),
    (False, 0),
    (False, 3),
])
def test_inMemory_Fellowship(update_labels, nb_tasks, dataset7c, dataset10c, dataset20c):
    fellow = Fellowship([dataset7c, dataset10c, dataset20c], update_labels=update_labels)
    continuum = InstanceIncremental(fellow, nb_tasks=nb_tasks)

    assert continuum.nb_tasks == 3


@pytest.mark.slow
@pytest.mark.parametrize("nb_tasks", [0, 3])
def test_MNIST_Fellowship_Instance_Incremental(nb_tasks, tmpdir):
    dataset = MNISTFellowship(data_path=tmpdir, train=True, download=True)
    dataset.get_data()
    continuum = InstanceIncremental(dataset, nb_tasks=nb_tasks)
    assert len(continuum) == 3


@pytest.mark.slow
def test_MNIST_Fellowship_nb_classes(tmpdir):
    dataset = MNISTFellowship(data_path=tmpdir, train=True, download=True)
    x, y, t = dataset.get_data()
    assert len(np.unique(y)) == 30
    dataset = MNISTFellowship(data_path=tmpdir, train=True, download=True, update_labels=False)
    x, y, t = dataset.get_data()
    assert len(np.unique(y)) == 10


@pytest.mark.slow
def test_MNIST_Fellowship(tmpdir):
    dataset = MNISTFellowship(data_path=tmpdir, train=True, download=True)
    dataset.get_data()
    continuum = ClassIncremental(dataset, increment=10)
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
    cl_dataset = Fellowship(
        datasets=[d(data_path=tmpdir, download=True, train=True) for d in list_datasets]
    )
    scenario = ClassIncremental(cl_dataset, increment=10)

    assert len(scenario) == nb_tasks
    for task_id, taskset in enumerate(scenario):
        classes = taskset.get_classes()

        # we check if all classes are here
        assert len(classes) == (classes.max() - classes.min() + 1)


@pytest.mark.slow
@pytest.mark.parametrize("list_datasets", [[MNIST, CIFAR10]])
def test_Fellowship_Dimension_Fail(tmpdir, list_datasets):
    cl_dataset = Fellowship(
        datasets=[d(data_path=tmpdir, download=True, train=True) for d in list_datasets]
    )

    # This does not work since CIFAR10 and MNIST data are not same shape
    with pytest.raises(ValueError):
        continuum = ClassIncremental(cl_dataset, increment=10)
