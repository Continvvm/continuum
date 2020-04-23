import numpy as np
import pytest
from clloader.scenarios import InstanceIncremental
from tests.test_classorder import InMemoryDatasetTest


@pytest.fixture
def numpy_data():

    nb_classes = 6
    nb_data = 100

    x_train = []
    y_train = []
    for i in range(nb_classes):
        x_train.append(np.ones((nb_data, 4, 4, 3), dtype=np.uint8) * i)
        y_train.append(np.ones(nb_data) * i)
    x_train = np.concatenate(x_train)
    y_train = np.concatenate(y_train)

    x_test = np.copy(x_train)
    y_test = np.copy(y_train)

    return (x_train, y_train.astype(int)), (x_test, y_test.astype(int))

# yapf: disable


def test_constant_class_number(numpy_data):
    train, test = numpy_data
    dummy = InMemoryDatasetTest(*train, *test)
    clloader = InstanceIncremental(dummy, 2)

    nb_classes = clloader.nb_classes

    for task_id, (train_dataset, test_dataset) in enumerate(clloader):
        assert nb_classes == len(np.unique(train_dataset.y))
