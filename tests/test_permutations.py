import numpy as np
import pytest
import torch
from torchvision.transforms import transforms

from continuum.scenarios import Permutations
from tests.test_classorder import InMemoryDatasetTest
from continuum.datasets import MNIST, CIFAR100


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
    if isinstance(seed, list):
        nb_tasks = len(seed) + 1

    scenario_1 = Permutations(cl_dataset=dummy, nb_tasks=nb_tasks, seed=seed)
    scenario_2 = Permutations(cl_dataset=dummy, nb_tasks=nb_tasks, seed=seed)

    previous_x = []
    if isinstance(seed, list):
        assert len(scenario_1) == len(scenario_2) == len(seed) + 1

    for task_id, (train_taskset_1, train_taskset_2) in enumerate(zip(scenario_1, scenario_2)):
        assert task_id < nb_tasks

        assert len(train_taskset_1) == len(train_taskset_2)
        indexes = list(range(len(train_taskset_1)))

        x_1, y_1, t_1 = train_taskset_1.get_samples(indexes)
        x_2, y_2, t_2 = train_taskset_2.get_samples(indexes)

        assert (x_1 == x_2).all()
        assert (y_1 == y_2).all()
        assert (t_1 == t_2).all()

        for x in previous_x:
            assert not (x == x_1).all()
        previous_x.append(x_1.clone())


@pytest.mark.slow
@pytest.mark.parametrize("shared_label_space", [True, False])
@pytest.mark.parametrize("dataset", [MNIST, CIFAR100])
def test_with_dataset(dataset, shared_label_space):
    dataset = dataset(data_path="./tests/Datasets", download=True, train=True)
    scenario = Permutations(cl_dataset=dataset, nb_tasks=5, seed=0, shared_label_space=shared_label_space)

    for task_id, taskset in enumerate(scenario):

        classes = taskset.get_classes()

        if shared_label_space:
            assert len(classes) == classes.max() + 1
        else:
            assert len(classes) == classes.max() + 1 - (task_id * len(classes))
