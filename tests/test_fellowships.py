import os
import copy

import pytest
import numpy as np
from torch.utils.data import DataLoader
import torchvision.transforms as trsf

from continuum.tasks import TaskType
from continuum.scenarios import ClassIncremental, InstanceIncremental, OnlineFellowship, create_subscenario
from continuum.datasets import (
    CIFAR10, CIFAR100, KMNIST, MNIST, CIFARFellowship, FashionMNIST, Fellowship, MNISTFellowship,
    InMemoryDataset, Fellowship, Core50
)

DATA_PATH = os.environ.get("CONTINUUM_DATA_PATH")


@pytest.fixture
def dataset7c():
    return InMemoryDataset(*gen_dataset(7, 0))


@pytest.fixture
def dataset10c():
    return InMemoryDataset(*gen_dataset(10, 1))


@pytest.fixture
def dataset20c():
    return InMemoryDataset(*gen_dataset(20, 2))


@pytest.fixture
def dataset20c_3channels():
    return InMemoryDataset(*gen_dataset_3channels(20, 2))


def gen_dataset(nb_classes, pixel_value):
    nb_items_per_class = 5

    x_train = np.ones((nb_items_per_class * nb_classes, 32, 32, 3)) * pixel_value
    y_train = []
    for i in range(nb_classes):
        y_train.append(np.ones(nb_items_per_class, dtype=np.int64) * i)
    y_train = np.concatenate(y_train)

    return (x_train, y_train)


def gen_dataset_3channels(nb_classes, pixel_value):
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


def test_Online_Fellowship(dataset7c, dataset10c, dataset20c):
    scenario = OnlineFellowship([dataset7c, dataset10c, dataset20c])
    for i, task_set in enumerate(scenario):
        if i == 0:
            assert task_set.nb_classes == 7
        if i == 1:
            assert task_set.nb_classes == 10
        if i == 2:
            assert task_set.nb_classes == 20

    assert scenario[0].nb_classes == 7
    assert scenario[1].nb_classes == 10
    assert scenario[2].nb_classes == 20


def test_Online_Fellowship_subscenarios(dataset7c, dataset10c, dataset20c):
    scenario = OnlineFellowship([dataset7c, dataset10c, dataset20c])
    sub_scenario = create_subscenario(scenario, np.arange(scenario.nb_tasks - 1))

    for task_set in sub_scenario:
        loader = DataLoader(task_set)
        for _ in loader:
            pass

    assert sub_scenario.nb_tasks == scenario.nb_tasks - 1

    task_order = np.arange(scenario.nb_tasks)
    np.random.shuffle(task_order)
    sub_scenario = create_subscenario(scenario, task_order)

    for task_set in sub_scenario:
        loader = DataLoader(task_set)
        for _ in loader:
            pass

    assert sub_scenario.nb_tasks == scenario.nb_tasks


@pytest.mark.parametrize("types,error", (
        [[TaskType.IMAGE_PATH], False],
        [[TaskType.H5, TaskType.IMAGE_PATH, TaskType.IMAGE_ARRAY, TaskType.TENSOR], False],
        [[TaskType.H5, TaskType.IMAGE_PATH, TaskType.IMAGE_ARRAY, TaskType.TENSOR, TaskType.SEGMENTATION], True],
        [[TaskType.H5, TaskType.IMAGE_PATH, TaskType.IMAGE_ARRAY, TaskType.TENSOR, TaskType.TEXT], True],
        [[TaskType.H5, TaskType.IMAGE_PATH, TaskType.IMAGE_ARRAY, TaskType.TENSOR, TaskType.OBJ_DETECTION], True],
        [[TaskType.SEGMENTATION, TaskType.OBJ_DETECTION], True],
        [[TaskType.SEGMENTATION], False],
))
def test_online_Fellowship_mixeddatatype(dataset10c, types, error):
    datasets = []
    for typ in types:
        d = copy.deepcopy(dataset10c)
        d._data_type = typ
        d._nb_classes = 10
        datasets.append(d)

    if error:
        with pytest.raises(ValueError):
            scenario = OnlineFellowship(datasets)
    else:
        scenario = OnlineFellowship(datasets)


@pytest.mark.slow
@pytest.mark.parametrize(
    "list_datasets", [
        ([MNIST, FashionMNIST]),
        ([KMNIST, MNIST, FashionMNIST]),
        ([CIFAR10, CIFAR100, KMNIST, MNIST, FashionMNIST]),
    ]
)
def test_online_Fellowship_inMemory(list_datasets):
    list_dict_args = {"data_path": DATA_PATH, "train": True, "download": False}

    list_instanciate_datasets = []
    for dataset in list_datasets:
        list_instanciate_datasets.append(dataset(**list_dict_args))

    scenario = OnlineFellowship(list_instanciate_datasets, update_labels=True)

    assert len(scenario) == len(list_datasets)
    tot_nb_classes = 0

    for task_id, taskset in enumerate(scenario):
        tot_nb_classes += taskset.nb_classes

        loader = DataLoader(taskset)
        _, _, _ = next(iter(loader))

    assert tot_nb_classes == scenario.nb_classes


@pytest.mark.slow
@pytest.mark.parametrize(
    "list_datasets", [
        ([Core50, CIFAR10])
    ]
)
def test_online_Fellowship_mix_path_array(list_datasets):
    list_dict_args = [{"data_path": DATA_PATH, "train": True, "download": False}] * len(list_datasets)

    list_instanciate_datasets = []
    for i, dataset in enumerate(list_datasets):
        list_instanciate_datasets.append(dataset(**list_dict_args[i]))

    scenario = OnlineFellowship(list_instanciate_datasets, update_labels=True)

    assert len(scenario) == len(list_datasets)
    tot_nb_classes = 0

    for task_id, taskset in enumerate(scenario):
        tot_nb_classes += taskset.nb_classes
        loader = DataLoader(taskset)
        _, _, _ = next(iter(loader))

    assert tot_nb_classes == scenario.nb_classes


@pytest.mark.parametrize(
    "transformations", [
        ([trsf.Resize(size=(16, 16)), trsf.ToTensor()]),  # single for all
        ([[trsf.ToTensor()], [trsf.ToTensor()], [trsf.ToTensor()]])  # one each
    ]
)
def test_online_Fellowship_transformation(dataset7c, dataset10c, dataset20c, transformations):
    scenario = OnlineFellowship([dataset7c, dataset10c, dataset20c], transformations=transformations)

    assert len(scenario) == 3
    tot_nb_classes = 0

    for task_id, taskset in enumerate(scenario):
        tot_nb_classes += taskset.nb_classes
        loader = DataLoader(taskset)
        _, _, _ = next(iter(loader))

    assert tot_nb_classes == scenario.nb_classes


def test_online_Fellowship_transformation2(dataset7c, dataset10c, dataset20c):
    sizes = [16, 24, 40]
    transformations = [[trsf.Resize(size=(sizes[0], sizes[0])), trsf.ToTensor()],
                       [trsf.Resize(size=(sizes[1], sizes[1])), trsf.ToTensor()],
                       [trsf.Resize(size=(sizes[2], sizes[2])), trsf.ToTensor()]]
    scenario = OnlineFellowship([dataset7c, dataset10c, dataset20c], transformations=transformations)

    for task_id, taskset in enumerate(scenario):
        loader = DataLoader(taskset)
        x, _, _ = next(iter(loader))
        assert x.shape[-1] == sizes[task_id]


@pytest.mark.parametrize("increment", [1, [7, 10, 20]])
def test_inMemory_keepLabels_Fellowship(increment, dataset7c, dataset10c, dataset20c):
    fellow = Fellowship([dataset7c, dataset10c, dataset20c], update_labels=False)

    x, y, t = fellow.get_data()
    assert len(np.unique(t)) == 3
    assert len(np.unique(y)) == 20

    if isinstance(increment, list):
        with pytest.raises(Exception):
            scenario = ClassIncremental(fellow, increment=increment)
    else:
        scenario = ClassIncremental(fellow, increment=increment)
        assert scenario.nb_classes == 20
        assert scenario.nb_tasks == 20


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
