import os
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as trsf

from continuum.scenarios import ContinualScenario
from continuum.datasets import H5Dataset, CIFAR100, Core50

DATA_PATH = os.environ.get("CONTINUUM_DATA_PATH")


def gen_data():
    x_ = np.random.randint(0, 255, size=(20, 32, 32, 3))
    y_ = []
    for i in range(10):
        y_.append(np.ones(2) * i)
    y_ = np.concatenate(y_)

    t_ = np.copy(y_) // 5

    return x_, y_.astype(int), t_.astype(int)


# yapf: disable
def test_creation_h5dataset():
    filename_h5 = "test_h5.hdf5"
    if os.path.exists(filename_h5):
        os.remove(filename_h5)

    x_, y_, t_ = gen_data()
    h5dataset = H5Dataset(x_, y_, t_, data_path=filename_h5)

    x_0, y_0, t_0 = h5dataset.get_data()

    data_indexes = np.where(t_ == 0)[0]
    assert isinstance(x_0, str) # x is only the path to the file
    assert len(y_0) == len(y_)
    assert len(t_0) == len(t_)

    os.remove(filename_h5)


def test_concatenate_h5dataset():
    filename_h5 = "test_h5.hdf5"
    if os.path.exists(filename_h5):
        os.remove(filename_h5)

    x_, y_, t_ = gen_data()
    h5dataset = H5Dataset(x_, y_, t_, data_path=filename_h5)
    h5dataset.add_data(x_, y_, t_)

    os.remove(filename_h5)


def test_h5dataset_ContinualScenario():
    filename_h5 = "test_h5.hdf5"
    if os.path.exists(filename_h5):
        os.remove(filename_h5)

    x_, y_, t_ = gen_data()
    h5dataset = H5Dataset(x_, y_, t_, data_path=filename_h5)
    h5dataset.add_data(x_, y_, t_)

    nb_task = len(np.unique(t_))
    scenario = ContinualScenario(h5dataset)

    assert scenario.nb_tasks == nb_task

    os.remove(filename_h5)


def test_h5dataset_ContinualScenario():
    filename_h5 = "test_h5.hdf5"
    if os.path.exists(filename_h5):
        os.remove(filename_h5)

    x_, y_, t_ = gen_data()
    h5dataset = H5Dataset(x_, y_, t_, data_path=filename_h5)
    h5dataset.add_data(x_, y_, t_)

    nb_task = len(np.unique(t_))
    scenario = ContinualScenario(h5dataset)

    assert scenario.nb_tasks == nb_task

    os.remove(filename_h5)


def test_h5dataset_loading():
    filename_h5 = "test_h5.hdf5"
    if os.path.exists(filename_h5):
        os.remove(filename_h5)

    x_, y_, t_ = gen_data()
    h5dataset = H5Dataset(x_, y_, t_, data_path=filename_h5)
    h5dataset.add_data(x_, y_, t_)

    nb_task = len(np.unique(t_))
    scenario = ContinualScenario(h5dataset)

    for task_set in scenario:
        loader = DataLoader(task_set)
        for _ in loader:
            pass

    assert scenario.nb_tasks == nb_task
    os.remove(filename_h5)


@pytest.mark.slow
def test_on_array_dataset():
    filename_h5 = "test_CIFAR100_h5.hdf5"
    if os.path.exists(filename_h5):
        os.remove(filename_h5)

    cl_dataset = CIFAR100(data_path=DATA_PATH,
                          download=False,
                          train=True,
                          labels_type="category",
                          task_labels="lifelong")
    # in practice the construction is part by part to reduce data load but here we do it at once
    x, y, t = cl_dataset.get_data()
    h5dataset = H5Dataset(x, y, t, data_path=filename_h5)

    scenario = ContinualScenario(h5dataset)

    for task_set in scenario:
        loader = DataLoader(task_set, batch_size=64)
        for x, y, t in loader:
            assert x.shape == torch.Size([64, 3, 32, 32])
            break

    assert scenario.nb_tasks == 5  # number of task of CIFAR100Lifelong
    os.remove(filename_h5)

# Not compatible at the moment (it is not really necessary to use h5 when images are referenced by a path.)
# @pytest.mark.slow
# def test_on_path_dataset():
#     filename_h5 = "test_CIFAR100_h5.hdf5"
#     if os.path.exists(filename_h5):
#         os.remove(filename_h5)
#
#     cl_dataset = Core50(data_path=DATA_PATH,
#                         download=False,
#                         train=True,
#                         scenario="domains",
#                         classification="category")
#     # in practice the construction is part by part to reduce data load but here we do it at once
#     x, y, t = cl_dataset.get_data()
#     h5dataset = H5Dataset(x, y, t, data_path=filename_h5)
#
#     normalize = trsf.Normalize(mean=[0.485, 0.456, 0.406],
#                                std=[0.229, 0.224, 0.225])
#     resize = trsf.Resize(size=(224, 224))
#     transform = trsf.Compose([resize, trsf.ToTensor(), normalize])
#     list_transform = [transform]
#
#     scenario = ContinualScenario(h5dataset, transformations=list_transform)
#
#     for task_set in scenario:
#         loader = DataLoader(task_set, batch_size=64)
#         for x, y, t in loader:
#             assert x.shape == torch.Size([64, 3, 32, 32])
#             break
#
#     assert scenario.nb_tasks == 5  # number of task of CIFAR100Lifelong
#     os.remove(filename_h5)
