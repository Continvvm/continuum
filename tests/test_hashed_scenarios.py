import numpy as np
import pytest
import os

from continuum.datasets import CIFAR10, CIFAR100, TinyImageNet200, InMemoryDataset
from continuum.scenarios import HashedScenario

DATA_PATH = os.environ.get("CONTINUUM_DATA_PATH")


@pytest.mark.slow
@pytest.mark.parametrize("hash_name",
                         ["Whash",
                          "DhashV",
                          "DhashH",
                          "PhashSimple",
                          "Phash",
                          "AverageHash",
                          "ColorHash"])  # , "CropResistantHash"]) # too long CropResistantHash
@pytest.mark.parametrize(("dataset", "shape"),
                         [(CIFAR10, [32, 32, 3]),
                          (CIFAR100, [32, 32, 3]),
                          (TinyImageNet200, [64, 64, 3])])
def test_visualization_HashedScenario(hash_name, dataset, shape):
    num_tasks = 5
    dataset = dataset(data_path=DATA_PATH, download=False, train=True)
    scenario = HashedScenario(cl_dataset=dataset,
                              hash_name=hash_name,
                              nb_tasks=num_tasks)

    folder = os.path.join(DATA_PATH, "tests/Samples/HashedScenario/")
    if not os.path.exists(folder):
        os.makedirs(folder)

    for task_id, taskset in enumerate(scenario):
        taskset.plot(path=folder,
                     title="HashedScenario_{}_{}_{}.jpg".format(type(dataset).__name__, hash_name, task_id),
                     nb_samples=100,
                     shape=shape)


def numpy_data():
    x_train = []
    y_train = []
    for i in range(10):
        x_train.append(np.random.randn(5, 10, 10, 3))
        y_train.append(np.ones(5) * i)
    x_train = np.concatenate(x_train)
    y_train = np.concatenate(y_train)

    return (x_train, y_train.astype(int))


@pytest.mark.parametrize("hash_name",
                         ["Whash",
                          "DhashV",
                          "DhashH",
                          "PhashSimple",
                          "Phash",
                          "AverageHash",
                          "ColorHash", "CropResistantHash"])
def test_HashedScenario_save_indexes(hash_name):
    num_tasks = 2
    x, y = numpy_data()
    dataset = InMemoryDataset(x, y, None, data_type="image_array")
    folder = os.path.join(DATA_PATH, "tests/Samples/HashedScenario/")
    if not os.path.exists(folder):
        os.makedirs(folder)

    filename_indexes = f"{folder}/{hash_name}.npy"

    # test save the indexes array
    scenario = HashedScenario(cl_dataset=dataset,
                              hash_name=hash_name,
                              nb_tasks=num_tasks,
                              filename_hash_indexes=filename_indexes)

    # test load the indexes array
    scenario = HashedScenario(cl_dataset=dataset,
                              hash_name=hash_name,
                              nb_tasks=num_tasks,
                              filename_hash_indexes=filename_indexes)

    # delete test indexes
    os.remove(filename_indexes)
