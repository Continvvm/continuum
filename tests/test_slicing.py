import numpy as np
import pytest

from continuum.datasets import InMemoryDataset
from continuum.scenarios import ClassIncremental, InstanceIncremental


def gen_data():
    """
    Generate training dataset.

    Args:
    """
    x_train = np.random.randint(0, 255, size=(20, 32, 32, 3))
    y_train = []
    for i in range(10):
        y_train.append(np.ones(2) * i)
    y_train = np.concatenate(y_train)

    x_test = np.random.randint(0, 255, size=(20, 32, 32, 3))
    y_test = np.copy(y_train)

    return (x_train, y_train), (x_test, y_test)


# yapf: disable

@pytest.mark.parametrize("index,classes", [
    (0, [0, 1]),
    (1, [2, 3]),
    (slice(0, 2, 1), [0, 1, 2, 3]),
    (slice(0, 2), [0, 1, 2, 3]),
    (slice(2), [0, 1, 2, 3]),
    (slice(1, 3), [2, 3, 4, 5]),
    (slice(0, 10, 2), [0, 1, 4, 5, 8, 9]),
    (-1, [8, 9]),
    (-5, [0, 1]),
    (slice(-1, -3, -1), [6, 7, 8, 9]),
    (-6, [8, 9]),  # full loop
    (-7, [6, 7]),  # full loop
    (-20, [0, 1]),  # full loop
])
def test_slicing_nc(index, classes):
    """
    Test to test_slicing dataset.

    Args:
        index: (int): write your description
        classes: (todo): write your description
    """
    train, test = gen_data()
    dummy = InMemoryDataset(*train)
    scenario = ClassIncremental(dummy, increment=2)
    taskset = scenario[index]
    targets = np.sort(np.unique(taskset._y))
    assert len(targets) == len(classes)
    assert (targets == np.array(classes)).all(), (targets, classes)


@pytest.mark.parametrize("start_index,classes", [
    (0, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
    (3, [6, 7, 8, 9]),
    (4, [8, 9]),
    (12, [])
])
def test_slicing_nc_no_end(start_index, classes):
    """
    Test for the test dataset.

    Args:
        start_index: (int): write your description
        classes: (todo): write your description
    """
    train, test = gen_data()
    dummy = InMemoryDataset(*train)
    scenario = ClassIncremental(dummy, increment=2)
    taskset = scenario[start_index:]
    targets = np.sort(np.unique(taskset._y))
    assert len(targets) == len(classes)
    assert (targets == np.array(classes)).all(), (targets, classes)


def test_slicing_nc_no_index():
    """
    Generate index ising dataset.

    Args:
    """
    train, test = gen_data()
    dummy = InMemoryDataset(*train)
    scenario = ClassIncremental(dummy, increment=2)
    taskset = scenario[:]
    targets = np.sort(np.unique(taskset._y))

    classes = list(range(10))

    assert len(targets) == len(classes)
    assert (targets == np.array(classes)).all(), (targets, classes)
