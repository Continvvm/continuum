import numpy as np
import pytest
from torch.utils.data import DataLoader

from continuum.tasks import TaskSet, split_train_val, concat


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
