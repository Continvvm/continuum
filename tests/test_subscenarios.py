import numpy as np
import pytest
import random
import string

from continuum.datasets import InMemoryDataset
from continuum.scenarios import ClassIncremental, ContinualScenario, create_subscenario


def gen_data():
    x_train = np.random.randint(0, 255, size=(20, 32, 32, 3))
    y_train = []
    for i in range(10):
        y_train.append(np.ones(2) * i)
    y_train = np.concatenate(y_train)

    x_test = np.random.randint(0, 255, size=(20, 32, 32, 3))
    y_test = np.copy(y_train)

    return (x_train, y_train), (x_test, y_test)


def gen_string():
    """"create random string. We plan to use them as it was some path"""
    len_string = 20
    y_train = []
    list_str = []
    for i in range(20):
        value = ''.join(random.SystemRandom().choice(string.ascii_letters + string.digits) for _ in range(len_string))
        list_str.append([value])
    x_train = np.concatenate(list_str)

    for i in range(10):
        y_train.append(np.ones(2) * i)
    y_train = np.concatenate(y_train)
    return x_train, y_train


@pytest.mark.parametrize("list_tasks", [
    np.arange(10),
    np.arange(5, 10),
    np.arange(3, 10, 2),
    np.arange(9, 0, -2),
    np.arange(0, 10, 2),
    list(np.arange(0, 10, 2)),
    list(np.arange(5, 10))
])
def test_slicing_list(list_tasks):
    train, test = gen_data()
    dummy = InMemoryDataset(*train)
    scenario = ClassIncremental(dummy, increment=1)
    subscenario = create_subscenario(scenario, list_tasks)
    assert subscenario.nb_tasks == len(list_tasks), print(f"{len(subscenario)} - vs - {len(list_tasks)}")


@pytest.mark.parametrize("list_tasks", [
    np.arange(10),
    np.arange(5, 10),
    np.arange(3, 10, 2),
    np.arange(9, 0, -2),
])
def test_slicing_list_path_array(list_tasks):
    x_train, y_train = gen_string()
    dummy = InMemoryDataset(x_train, y_train, data_type="image_path")
    scenario = ClassIncremental(dummy, increment=1)
    subscenario = create_subscenario(scenario, list_tasks)
    assert subscenario.nb_tasks == len(list_tasks), print(f"{len(subscenario)} - vs - {len(list_tasks)}")
