import os
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from continuum.tasks import split_train_val, TaskType
from continuum.datasets import InMemoryDataset, H5Dataset, CIFAR100
from continuum.scenarios import ContinualScenario


def gen_data():
    x_ = np.random.randint(0, 255, size=(20, 32, 32, 3))
    y_ = []
    for i in range(10):
        y_.append(np.ones(2) * i)
    y_ = np.concatenate(y_)

    t_ = np.copy(y_)//5

    return x_, y_.astype(int), t_.astype(int)


# yapf: disable

def test_creation_h5dataset():
    x_, y_, t_ = gen_data()
    h5dataset = H5Dataset(x_, y_, t_, data_path="test_h5", data_type=TaskType.IMAGE_ARRAY)


    x_0, y_0, t_0 = h5dataset.get_task_data(ind_task = 0)

    data_indexes = np.where(t_==0)[0]
    assert x_0.shape == x_[data_indexes].shape
    assert len(y_0) == len(y_[data_indexes])
    assert len(t_0) == len(t_[data_indexes])

def test_concatenate_h5dataset():
    x_, y_, t_ = gen_data()
    h5dataset = H5Dataset(x_, y_, t_, data_path="test_h5", data_type=TaskType.IMAGE_ARRAY)
    h5dataset.add_data(x_, y_, t_)


def test_h5dataset_ContinualScenario():
    x_, y_, t_ = gen_data()
    h5dataset = H5Dataset(x_, y_, t_, data_path="test_h5", data_type=TaskType.IMAGE_ARRAY)
    h5dataset.add_data(x_, y_, t_)

    nb_task = len(np.unique(t_))
    scenario = ContinualScenario(h5dataset)

    assert scenario.nb_tasks == nb_task

def test_h5dataset_ContinualScenario():
    x_, y_, t_ = gen_data()
    h5dataset = H5Dataset(x_, y_, t_, data_path="test_h5", data_type=TaskType.IMAGE_ARRAY)
    h5dataset.add_data(x_, y_, t_)

    nb_task = len(np.unique(t_))
    scenario = ContinualScenario(h5dataset)

    assert scenario.nb_tasks == nb_task

def test_h5dataset_loading():
    x_, y_, t_ = gen_data()
    h5dataset = H5Dataset(x_, y_, t_, data_path="test_h5", data_type=TaskType.IMAGE_ARRAY)
    h5dataset.add_data(x_, y_, t_)

    nb_task = len(np.unique(t_))
    scenario = ContinualScenario(h5dataset)

    for task_set in scenario:
        loader = DataLoader(task_set)
        for _ in loader:
            pass

    assert scenario.nb_tasks == nb_task

DATA_PATH = os.environ.get("CONTINUUM_DATA_PATH")

@pytest.mark.slow
def test_on_dataset():
    cl_dataset = CIFAR100(data_path=DATA_PATH,
                          download=False,
                          train=True,
                          labels_type="category",
                          task_labels="lifelong")
    # in practice the construction is part by part to reduce data load but here we do it at once
    x, y, t = cl_dataset.get_data()
    h5dataset = H5Dataset(x, y, t, data_path="test_CIFAR100_h5", data_type=TaskType.IMAGE_ARRAY)

    scenario = ContinualScenario(h5dataset)

    for task_set in scenario:
        loader = DataLoader(task_set, batch_size=64)
        for x, y, t in loader:
            assert x.shape == torch.Size([64, 3, 32, 32])
            break

    assert scenario.nb_tasks == 5 # number of task of CIFAR100Lifelong

