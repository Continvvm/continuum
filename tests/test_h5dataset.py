import os
import time
import h5py
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as trsf

from continuum.scenarios import ContinualScenario, ClassIncremental, Permutations
from continuum.datasets import H5Dataset, CIFAR100, MNIST
from continuum.tasks.h5_task_set import H5TaskSet

DATA_PATH = os.environ.get("CONTINUUM_DATA_PATH")


@pytest.fixture
def data():
    x_ = np.random.randint(0, 255, size=(20, 32, 32, 3))
    y_ = []
    for i in range(10):
        y_.append(np.ones(2) * i)
    y_ = np.concatenate(y_)

    t_ = np.copy(y_) // 5

    return x_, y_.astype(int), t_.astype(int)


# yapf: disable
def test_creation_h5dataset(data, tmpdir):
    filename_h5 = os.path.join(tmpdir, "test_h5.hdf5")

    x_, y_, t_ = data
    h5dataset = H5Dataset(x_, y_, t_, data_path=filename_h5)

    x_0, y_0, t_0 = h5dataset.get_data()

    assert isinstance(x_0, str)  # x is only the path to the file
    assert len(y_0) == len(y_)
    assert len(t_0) == len(t_)


def test_concatenate_h5dataset(data, tmpdir):
    filename_h5 = os.path.join(tmpdir, "test_h5.hdf5")

    x_, y_, t_ = data
    h5dataset = H5Dataset(x_, y_, t_, data_path=filename_h5)
    h5dataset.add_data(x_, y_, t_)

    assert len(h5dataset.get_class_vector()) == 2 * len(y_)


def test_h5dataset_ContinualScenario(data, tmpdir):
    filename_h5 = os.path.join(tmpdir, "test_h5.hdf5")

    x_, y_, t_ = data
    h5dataset = H5Dataset(x_, y_, t_, data_path=filename_h5)

    nb_task = len(np.unique(t_))
    scenario = ContinualScenario(h5dataset)

    assert scenario.nb_tasks == nb_task

    data_indexes = np.where(t_ == 0)[0]
    assert len(data_indexes) == len(scenario[0])


def test_h5dataset_add_data(data, tmpdir):
    filename_h5 = os.path.join(tmpdir, "test_h5.hdf5")

    x_, y_, t_ = data
    h5dataset = H5Dataset(x_, y_, t_, data_path=filename_h5)
    h5dataset.add_data(x_, y_, t_)

    nb_task = len(np.unique(t_))
    scenario = ContinualScenario(h5dataset)

    assert scenario.nb_tasks == nb_task


def test_h5dataset_IncrementalScenario(data, tmpdir):
    filename_h5 = os.path.join(tmpdir, "test_h5.hdf5")

    x_, y_, t_ = data
    nb_task = 2
    h5dataset = H5Dataset(x_, y_, None, data_path=filename_h5)

    scenario = ClassIncremental(h5dataset, nb_tasks=nb_task)

    assert scenario.nb_tasks == nb_task

    tot_len = 0
    for task_set in scenario:
        tot_len += len(task_set)
        loader = DataLoader(task_set)
        for _ in loader:
            pass

    assert tot_len == len(y_)


def test_h5dataset_loading(data, tmpdir):
    filename_h5 = os.path.join(tmpdir, "test_h5.hdf5")

    x_, y_, t_ = data
    h5dataset = H5Dataset(x_, y_, t_, data_path=filename_h5)
    h5dataset.add_data(x_, y_, t_)

    nb_task = len(np.unique(t_))
    scenario = ContinualScenario(h5dataset)

    for task_set in scenario:
        loader = DataLoader(task_set)
        for _ in loader:
            pass

    assert scenario.nb_tasks == nb_task


def test_h5dataset_to_taskset(data, tmpdir):
    filename_h5 = os.path.join(tmpdir, "test_h5.hdf5")

    x_, y_, t_ = data
    h5dataset = H5Dataset(x_, y_, t_, data_path=filename_h5)
    task_set = h5dataset.to_taskset()
    loader = DataLoader(task_set)
    for _ in loader:
        pass


@pytest.mark.slow
def test_time(tmpdir):
    global DATA_PATH
    cl_dataset = CIFAR100(data_path=DATA_PATH,
                          download=False,
                          train=True,
                          labels_type="category",
                          task_labels="lifelong")
    # in practice the construction is part by part to reduce data load but here we do it at once
    x, y, t = cl_dataset.get_data()
    h5_filename = os.path.join(tmpdir, "test_time_h5.hdf5")
    h5dataset = H5Dataset(x, y, t, data_path=h5_filename)

    task_set = H5TaskSet(h5_filename, y=h5dataset.get_class_vector(), t=h5dataset.get_task_indexes(), trsf=None)

    start = time.time()
    for i in range(10000):
        a = task_set[5]
    end = time.time()
    print(f"normal __getitem__ {end - start}")

    start = time.time()
    with h5py.File(h5_filename, 'r') as hf:
        for i in range(10000):
            x = hf['x'][5]
            y = hf['y'][5]
            if 't' in hf.keys():
                t = hf['t'][5]
            else:
                t = -1
    end = time.time()
    print(f"open only once __getitem__ {end - start}")


@pytest.mark.slow
def test_on_array_dataset_incremental(tmpdir):
    filename_h5 = os.path.join(tmpdir, "test_CIFAR100_h5.hdf5")

    nb_tasks = 10

    cl_dataset = CIFAR100(data_path=DATA_PATH,
                          download=False,
                          train=True)
    # in practice the construction is part by part to reduce data load but here we do it at once
    x, y, t = cl_dataset.get_data()
    h5dataset = H5Dataset(x, y, t, data_path=filename_h5)

    scenario = ClassIncremental(h5dataset, nb_tasks=nb_tasks)

    for task_set in scenario:
        loader = DataLoader(task_set, batch_size=64)
        for x, y, t in loader:
            assert x.shape == torch.Size([64, 3, 32, 32])
            break

    assert scenario.nb_tasks == nb_tasks  # number of task of CIFAR100Lifelong


@pytest.mark.slow
def test_on_array_dataset(tmpdir):
    filename_h5 = os.path.join(tmpdir, "test_CIFAR100_h5.hdf5")

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

# Not compatible at the moment (it is less necessary to use h5 with transform incremental scenarios.)
# @pytest.mark.slow
# def test_on_transform_scenario():
#     filename_h5 = "test_permutation.hdf5"
#     if os.path.exists(filename_h5):
#         os.remove(filename_h5)
#
#     cl_dataset = MNIST(data_path=DATA_PATH,
#                           download=False,
#                           train=True)
#     # in practice the construction is part by part to reduce data load but here we do it at once
#     x, y, t = cl_dataset.get_data()
#     h5dataset = H5Dataset(x, y, t, data_path=filename_h5)
#
#     scenario = Permutations(h5dataset, nb_tasks=3, shared_label_space=True)
#
#     for task_set in scenario:
#         loader = DataLoader(task_set, batch_size=64)
#         for x, y, t in loader:
#             break
#
#     # SECOND TEST WITH A LABEL TRANSFORMATION
#
#     scenario = Permutations(h5dataset, nb_tasks=3, shared_label_space=False)
#
#     for task_set in scenario:
#         loader = DataLoader(task_set, batch_size=64)
#         for x, y, t in loader:
#             break
#
#     assert scenario.nb_tasks == 5  # number of task of CIFAR100Lifelong
#     os.remove(filename_h5)

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
