import numpy as np
import pytest
import os

from continuum.datasets import CIFAR10, CIFAR100, TinyImageNet200, InMemoryDataset
from continuum.scenarios import HashedScenario
from continuum.tasks import TaskType


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
@pytest.mark.parametrize(("dataset", "shape", "split_task"),
                         [(CIFAR10, [32, 32, 3], "auto"),
                          (CIFAR100, [32, 32, 3], "auto"),
                          (TinyImageNet200, [64, 64, 3], "auto"),
                          (CIFAR10, [32, 32, 3], "balanced"),
                          (CIFAR100, [32, 32, 3], "balanced"),
                          (TinyImageNet200, [64, 64, 3], "balanced")])
def test_visualization_HashedScenario(hash_name, dataset, shape, split_task):
    if split_task == "balanced":
        num_tasks = 5
    else:
        num_tasks = None

    dataset = dataset(data_path=DATA_PATH, download=False, train=True)
    scenario = HashedScenario(cl_dataset=dataset,
                              hash_name=hash_name,
                              nb_tasks=num_tasks,
                              split_task=split_task)

    assert scenario.nb_tasks > 1

    folder = "tests/samples/hashed_scenario/"
    if not os.path.exists(folder):
        os.makedirs(folder)

    # test default parameters
    for task_id, taskset in enumerate(scenario):
        taskset.plot(path=folder,
                     title="{}_HashedScenario_{}_{}_{}.jpg".format(split_task,
                                                                   type(dataset).__name__,
                                                                   hash_name,
                                                                   task_id),
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
                          "ColorHash"])  # , "CropResistantHash"
def test_HashedScenario_save_indexes(tmpdir, hash_name):
    num_tasks = 2
    x, y = numpy_data()
    dataset = InMemoryDataset(x, y, None, data_type=TaskType.IMAGE_ARRAY)

    filename_indexes = os.path.join(tmpdir, f"{hash_name}.npy")
    if os.path.exists(filename_indexes):
        os.remove(filename_indexes)

    if os.path.exists(filename_indexes):
        AssertionError(f"{filename_indexes} should have been delete.")

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


@pytest.mark.parametrize("hash_name",
                         ["Whash",
                          "DhashV",
                          "DhashH",
                          "PhashSimple",
                          "Phash",
                          "AverageHash",
                          "ColorHash"])  # , "CropResistantHash"
def test_HashedScenario_automatic_task_number(hash_name):
    x, y = numpy_data()
    dataset = InMemoryDataset(x, y, None, data_type=TaskType.IMAGE_ARRAY)

    # test when nb_tasks is set to None
    scenario = HashedScenario(cl_dataset=dataset,
                              hash_name=hash_name,
                              nb_tasks=None,
                              split_task="auto")

    if scenario.nb_tasks is None or scenario.nb_tasks < 2:
        AssertionError("nb_tasks should have been set automatically to more than one")
