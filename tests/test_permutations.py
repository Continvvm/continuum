import numpy as np
import pytest
from continuum.scenarios import Permutations
from tests.test_classorder import InMemoryDatasetTest
from torchvision.transforms import transforms

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

'''
Test the initialization with three tasks
'''
def test_init(numpy_data):
    train, test = numpy_data
    dummy = InMemoryDatasetTest(*train)
    clloader = Permutations(cl_dataset=dummy, nb_tasks=3, seed=0)


    for task_id, train_dataset in enumerate(clloader):
        assert task_id < 3
        continue

