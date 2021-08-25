import numpy as np
import pytest
import os

from continuum.datasets import CIFAR10, CIFAR100, TinyImageNet200
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
                          "ColorHash"]) #, "CropResistantHash"]) # too long CropResistantHash
@pytest.mark.parametrize("dataset,shape",
                         [(CIFAR10, [32, 32, 3]),
                         (CIFAR100, [32, 32, 3]),
                         (TinyImageNet200, [64, 64, 3])])
def test_visualization_HashedScenario(hash_name, dataset, shape):
    num_tasks = 5
    dataset = dataset(data_path=DATA_PATH, download=False, train=True)
    scenario = HashedScenario(cl_dataset=dataset,
                              hash_name=hash_name,
                              nb_tasks=num_tasks,
                              data_shape=shape)

    folder = os.path.join(DATA_PATH, "tests/Samples/HashedScenario/")
    if not os.path.exists(folder):
        os.makedirs(folder)

    for task_id, taskset in enumerate(scenario):
        taskset.plot(path=folder,
                     title="HashedScenario_{}_{}_{}.jpg".format(type(dataset).__name__, hash_name, task_id),
                     nb_samples=100,
                     shape=shape)
