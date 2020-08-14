import numpy as np
import pytest
import torch
from torchvision.transforms import transforms

from continuum.scenarios import Permutations
from tests.test_classorder import InMemoryDatasetTest


@pytest.fixture
def numpy_data():
    nb_classes = 6
    nb_data = 100

    x_train = []
    y_train = []
    x_train.append(
        np.array([np.random.randint(100, size=(2, 2, 3)).astype(dtype=np.uint8)] * nb_data)
    )
    y_train.append(np.random.randint(nb_classes, size=(nb_data)))
    x_train = np.concatenate(x_train)
    y_train = np.concatenate(y_train)

    x_test = np.copy(x_train)
    y_test = np.copy(y_train)

    return (x_train, y_train.astype(int)), (x_test, y_test.astype(int))


'''
Test the initialization with three tasks
'''


@pytest.mark.parametrize("seed", [0, 42, 1664, [0, 1, 2], [0, 1, 2, 3]])
def test_init(numpy_data, seed):
    train, test = numpy_data
    dummy = InMemoryDatasetTest(*train)

    nb_tasks = 3

    clloader_1 = Permutations(cl_dataset=dummy, nb_tasks=nb_tasks, seed=seed)
    clloader_2 = Permutations(cl_dataset=dummy, nb_tasks=nb_tasks, seed=seed)

    if isinstance(seed, list):
        assert len(clloader_1) == len(clloader_2) == len(seed) + 1
        nb_tasks = len(seed) + 1

    for task_id, (train_dataset_1, train_dataset_2) in enumerate(zip(clloader_1, clloader_2)):
        assert task_id < nb_tasks

        assert len(train_dataset_1) == len(train_dataset_2)
        indexes = list(range(len(train_dataset_1)))

        x_1, y_1, t_1 = train_dataset_1.get_samples(indexes)
        x_2, y_2, t_2 = train_dataset_2.get_samples(indexes)

        assert (x_1 == x_2).all()
        assert (y_1 == y_2).all()
        assert (t_1 == t_2).all()
