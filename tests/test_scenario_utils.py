import os
import numpy as np
import pytest
import random
import string

import torch.nn as nn

from continuum.datasets import InMemoryDataset, MNIST
from torchvision import transforms
from continuum.scenarios import ClassIncremental, create_subscenario, encode_scenario
from continuum.tasks import TaskType

DATA_PATH = os.environ.get("CONTINUUM_DATA_PATH")


def gen_data():
    x_ = np.random.randint(0, 255, size=(20, 32, 32, 3))
    y_ = []
    for i in range(10):
        y_.append(np.ones(2) * i)
    y_ = np.concatenate(y_)

    t_ = np.copy(y_) // 5

    return x_, y_.astype(int), t_.astype(int)


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
    train = gen_data()
    dummy = InMemoryDataset(*train)
    scenario = ClassIncremental(dummy, increment=1)
    subscenario = create_subscenario(scenario, list_tasks)
    assert subscenario.nb_tasks == len(list_tasks), print(f"{len(subscenario)} - vs - {len(list_tasks)}")


@pytest.mark.parametrize("list_tasks", [
    np.arange(10),
    np.arange(5, 10),
    np.arange(3, 10, 2),
    np.arange(9, 0, -2),
    np.arange(0, 10, 2),
    list(np.arange(0, 10, 2)),
    list(np.arange(5, 10))
])
def test_sequence_transforms(list_tasks):
    x_train, y_train, t_train = gen_data()
    dummy = InMemoryDataset(x_train, y_train, t_train, data_type=TaskType.IMAGE_PATH)

    nb_task = len(np.unique(y_train))
    list_trsfs = []
    for _ in range(nb_task):
        list_trsfs.append([transforms.RandomAffine(degrees=[0, 90])])

    scenario = ClassIncremental(dummy, increment=1, transformations=list_trsfs)
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
    dummy = InMemoryDataset(x_train, y_train, data_type=TaskType.IMAGE_PATH)
    scenario = ClassIncremental(dummy, increment=1)
    subscenario = create_subscenario(scenario, list_tasks)
    assert subscenario.nb_tasks == len(list_tasks), print(f"{len(subscenario)} - vs - {len(list_tasks)}")


def test_encode_scenario():
    filename_h5 = "test_encode_scenario.hdf5"
    if os.path.exists(filename_h5):
        os.remove(filename_h5)

    train = gen_data()
    x, y, t = train
    x = x.reshape(-1, 32 * 32 * 3)

    dummy = InMemoryDataset(x, y, t)
    scenario = ClassIncremental(dummy, increment=1)

    model = nn.Sequential(nn.Flatten(), nn.Linear(32 * 32 * 3, 50))
    encoded_scenario = encode_scenario(model=model,
                                       scenario=scenario,
                                       batch_size=64,
                                       filename=filename_h5)

    assert scenario.nb_tasks == encoded_scenario.nb_tasks
    assert len(scenario[0]) == len(encoded_scenario[0])

    os.remove(filename_h5)


def test_encode_scenario_inference_fct():
    filename_h5 = "test_encode_scenario.hdf5"
    if os.path.exists(filename_h5):
        os.remove(filename_h5)

    train = gen_data()

    dummy = InMemoryDataset(*train)
    scenario = ClassIncremental(dummy, increment=1)

    model = nn.Linear(32 * 32 * 3, 50)
    inference_fct = lambda model, x: model(x.view(-1, 32 * 32 * 3))

    encoded_scenario = encode_scenario(model=model,
                                       scenario=scenario,
                                       batch_size=64,
                                       filename=filename_h5,
                                       inference_fct=inference_fct)

    assert scenario.nb_tasks == encoded_scenario.nb_tasks
    assert len(scenario[0]) == len(encoded_scenario[0])

    assert encoded_scenario[0][0][0].shape[0] == 50

    os.remove(filename_h5)


@pytest.mark.slow
def test_encode_scenario_MNIST():
    filename_h5 = "test_encode_scenario.hdf5"
    if os.path.exists(filename_h5):
        os.remove(filename_h5)

    dataset = MNIST(data_path=DATA_PATH,
                    download=False,
                    train=True)
    scenario = ClassIncremental(dataset, increment=2)

    model = nn.Linear(28 * 28, 50)
    inference_fct = lambda model, x: model(x.view(-1, 28 * 28))

    encoded_scenario = encode_scenario(model=model,
                                       scenario=scenario,
                                       batch_size=264,
                                       filename=filename_h5,
                                       inference_fct=inference_fct)

    assert scenario.nb_tasks == encoded_scenario.nb_tasks

    for encoded_taskset, taskset in zip(encoded_scenario, scenario):
        assert len(encoded_taskset) == len(taskset)

    assert encoded_scenario[0][0][0].shape[0] == 50

    os.remove(filename_h5)
