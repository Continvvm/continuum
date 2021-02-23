import numpy as np
import pytest
from torch.utils.data import DataLoader

from continuum.tasks import split_train_val
from continuum.datasets import InMemoryDataset
from continuum.scenarios import ClassIncremental


def gen_data():
    x_train = np.random.randint(0, 255, size=(20, 32, 32, 3))
    y_train = []
    for i in range(10):
        y_train.append(np.ones(2) * i)
    y_train = np.concatenate(y_train)

    x_test = np.random.randint(0, 255, size=(20, 32, 32, 3))
    y_test = np.copy(y_train)

    return (x_train, y_train), (x_test, y_test)


# this function create data with a mismatch between x and y shape
def gen_bad_data():
    nb_classes = 6
    nb_data_x = 10
    nb_data_y = 100

    x_train = []
    y_train = []
    for i in range(nb_classes):
        x_train.append(np.random.randint(100, size=(nb_data_x, 2, 2, 3)).astype(dtype=np.uint8))
        y_train.append(np.ones(nb_data_y) * i)
    x_train = np.concatenate(x_train)
    y_train = np.concatenate(y_train)

    x_test = np.copy(x_train)
    y_test = np.copy(y_train)

    return (x_train, y_train.astype(int)), (x_test, y_test.astype(int))


# yapf: disable

@pytest.mark.parametrize("increment,initial_increment,nb_tasks", [
    (2, 0, 5),
    (5, 0, 2),
    (1, 5, 6),
    (2, 4, 4),
    ([5, 1, 1, 3], 0, 4)
])
def test_increments(increment, initial_increment, nb_tasks):
    train, test = gen_data()
    dummy = InMemoryDataset(*train)
    scenario = ClassIncremental(dummy, increment=increment, initial_increment=initial_increment)

    assert scenario.nb_tasks == nb_tasks
    seen_tasks = 0

    for task_id, taskset in enumerate(scenario):
        seen_tasks += 1

        if isinstance(increment, list):
            max_class = sum(increment[:task_id + 1])
            min_class = sum(increment[:task_id])
        elif initial_increment:
            max_class = initial_increment + increment * task_id
            min_class = initial_increment + increment * (task_id - 1) if task_id > 0 else 0
        else:
            max_class = increment * (task_id + 1)
            min_class = increment * task_id

        for _ in DataLoader(taskset):
            pass

        assert np.max(taskset._y) == max_class - 1
        assert np.min(taskset._y) == min_class
    assert seen_tasks == nb_tasks


def test_bad_data():
    train, test = gen_bad_data()
    with pytest.raises(ValueError):
        dummy = InMemoryDataset(*train)


@pytest.mark.parametrize("val_split", [0, 0.1, 0.5, 0.8, 1.0])
def test_split_train_val(val_split):
    train, test = gen_data()
    dummy = InMemoryDataset(*train)
    scenario = ClassIncremental(dummy, increment=5)

    for taskset in scenario:
        train_taskset, val_taskset = split_train_val(taskset, val_split=val_split)
        assert int(val_split * len(taskset)) == len(val_taskset)
        assert len(val_taskset) + len(train_taskset) == len(taskset)
