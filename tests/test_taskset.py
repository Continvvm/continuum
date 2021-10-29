import numpy as np
import pytest
from torchvision import transforms
from continuum.datasets import InMemoryDataset
from continuum.tasks import TaskSet, concat, split_train_val
from torch.utils.data import DataLoader
import torch

from continuum.datasets import InMemoryDataset
from continuum.tasks import TaskSet, concat, split_train_val, get_balanced_sampler


@pytest.mark.parametrize("log", [False, True])
def test_sampler_function(log):
    np.random.seed(1)
    torch.manual_seed(1)

    x = np.random.rand(100, 2, 2, 3)
    y = np.ones((100,), dtype=np.int64)
    y[0] = 0
    t = np.ones((100,))

    taskset = TaskSet(x, y, t, None)
    sampler = get_balanced_sampler(taskset, log=log)

    loader = DataLoader(taskset, sampler=sampler, batch_size=1)
    nb_0 = 0
    for x, y, t in loader:
        if 0 in y:
            nb_0 += 1
    assert nb_0 > 1


@pytest.mark.parametrize("nb_classes", [2, 3, 5])
def test_target_trsf(nb_classes):
    x = np.random.rand(10, 2, 2, 3)
    y = np.arange(10)
    t = np.ones((10,))

    target_trsf = transforms.Lambda(lambda x: x % nb_classes)
    tasket = TaskSet(x, y, t, None, target_trsf=target_trsf)

    assert tasket.nb_classes == nb_classes, print("target transform not applied in get_classes")


    loader = DataLoader(tasket)
    for x, y, t in loader:
        pass

@pytest.mark.parametrize("nb_others", [1, 2])
def test_concat_function(nb_others):
    x = np.random.rand(10, 2, 2, 3)
    y = np.ones((10,))
    t = np.ones((10,))

    task_sets = [
        TaskSet(np.copy(x), np.copy(y), np.copy(t), None) for _ in range(nb_others)
    ]

    concatenation = concat(task_sets)
    assert len(concatenation) == nb_others * 10
    loader = DataLoader(concatenation)
    for x, y, t in loader:
        pass


@pytest.mark.parametrize("nb_others", [0, 1, 2])
def test_concat_method(nb_others):
    x = np.random.rand(10, 2, 2, 3)
    y = np.ones((10,))
    t = np.ones((10,))

    base_set = TaskSet(x, y, t, None)
    initial_len = len(base_set)

    others = [
        TaskSet(np.copy(x), np.copy(y), np.copy(t), None) for _ in range(nb_others)
    ]
    base_set.concat(*others)
    assert len(base_set) == initial_len + nb_others * initial_len
    loader = DataLoader(base_set)
    for x, y, t in loader:
        pass


@pytest.mark.parametrize("val_split,nb_val", [(0., 0), (0.1, 1), (0.8, 8), (0.99, 9), (1.0, 10)])
def test_split_train_val(val_split, nb_val):
    x = np.random.rand(10, 2, 2, 3)
    y = np.ones((10,))
    t = np.ones((10,))

    base_set = TaskSet(x, y, t, None)

    train_set, val_set = split_train_val(base_set, val_split)
    assert len(val_set) == nb_val
    assert len(train_set) + len(val_set) == len(base_set)


def test_split_train_val_loading():
    x = np.random.rand(10, 2, 2, 3)
    y = np.ones((10,))
    t = np.ones((10,))

    base_set = TaskSet(x, y, t, None)

    train_set, val_set = split_train_val(base_set, 0.2)

    for task_set in (train_set, val_set):
        loader = DataLoader(task_set, batch_size=32)
        for x, y, t in loader:
            pass


@pytest.mark.parametrize("nb_samples", [1, 5, 10])
def test_get_random_samples(nb_samples):
    x = np.ones((10, 2, 2, 3))
    y = np.ones((10,))
    t = np.ones((10,))

    base_set = TaskSet(x, y, t, None)

    base_set.get_random_samples(nb_samples=nb_samples)


@pytest.mark.parametrize("nb_samples", [1, 5, 10])
def test_get_raw_samples(nb_samples):
    x = np.ones((10, 2, 2, 3))
    y = np.ones((10,))
    t = np.ones((10,))

    base_set = TaskSet(x, y, t, None)

    data, y_, t_ = base_set.get_raw_samples(indexes=range(nb_samples))

    assert (x[:nb_samples] == data).all()
    assert (y[:nb_samples] == y_).all()
    assert (t[:nb_samples] == t_).all()


def test_continuum_to_pytorch_dataset():
    x_train = np.random.randint(0, 255, size=(20, 32, 32, 3))
    y_train = []
    for i in range(10):
        y_train.append(np.ones(2) * i)
    y_train = np.concatenate(y_train)

    continuum_dataset = InMemoryDataset(x_train, y_train)
    task_set = continuum_dataset.to_taskset()

    loader = DataLoader(task_set, batch_size=32)

    c = 0
    for x, y, _ in loader:
        pass
