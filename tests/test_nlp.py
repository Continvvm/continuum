import numpy as np
import pytest

from continuum.datasets import InMemoryDataset
from continuum.scenarios import ClassIncremental, InstanceIncremental


@pytest.fixture
def numpy_data():
    nb_classes = 6
    nb_tasks = 3
    nb_data = 100

    x_train = []
    y_train = []
    t_train = []
    for i in range(nb_tasks):
        for j in range(nb_classes):
            for _ in range(nb_data):
                x_train.append(np.array(["hello world"]))
            y_train.append(np.ones(nb_data) * j)
            t_train.append(np.ones(nb_data) * i)

    x_train = np.concatenate(x_train)
    y_train = np.concatenate(y_train)
    t_train = np.concatenate(t_train)

    x_test = np.copy(x_train)
    y_test = np.copy(y_train)
    t_test = np.copy(t_train)

    return (x_train, y_train.astype(int), t_train), (x_test, y_test.astype(int), t_test)


def test_nlp_class_incremental(numpy_data):
    train, test = numpy_data

    x_train, y_train, t_train = train

    dummy = InMemoryDataset(x_train, y_train, data_type="text")

    clloader = ClassIncremental(dummy, increment=2)

    assert len(clloader) == clloader.nb_tasks == 3
    assert clloader.nb_classes == 6

    for _ in clloader:
        pass


def test_nlp_instance_incremental(numpy_data):
    train, test = numpy_data

    x_train, y_train, t_train = train

    dummy = InMemoryDataset(
        x_train, y_train, t_=t_train, data_type="text"
    )

    clloader = InstanceIncremental(dummy)

    assert len(clloader) == clloader.nb_tasks == 3
    assert clloader.nb_classes == 6

    for _ in clloader:
        pass
