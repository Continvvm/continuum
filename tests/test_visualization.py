import numpy as np
import pytest
import os

from continuum.datasets import MNIST, CIFAR10, CIFAR100, KMNIST, FashionMNIST, TinyImageNet200
from continuum.scenarios import Rotations
from continuum.scenarios import Permutations
from continuum.scenarios import ClassIncremental
from continuum.datasets import MNISTFellowship

DATA_PATH = os.environ.get("CONTINUUM_DATA_PATH")

@pytest.mark.slow
@pytest.mark.parametrize("dataset, name, shape", [
                                                  (CIFAR100, "CIFAR100", [32, 32, 3])])
def test_visualization_ClassIncremental(dataset, name, shape):
    increment = 2
    if name == "CIFAR100":
        increment = 20
    if name == "TinyImageNet200":
        increment = 40
    scenario = ClassIncremental(cl_dataset=dataset(data_path=DATA_PATH, download=True, train=True),
                                increment=increment)

    folder = os.path.join(DATA_PATH, "tests/Samples/ClassIncremental/")
    if not os.path.exists(folder):
        os.makedirs(folder)

    for task_id, taskset in enumerate(scenario):
        taskset.plot(path=folder,
                     title="{}_ClassIncremental_{}.jpg".format(name, task_id),
                     nb_samples=100,
                     shape=shape)


'''
Test the visualization with three tasks for rotations tasks
'''
@pytest.mark.slow
def test_visualization_rotations():
    scenario = Rotations(cl_dataset=MNIST(data_path=DATA_PATH, download=True, train=True),
                         nb_tasks=3,
                         list_degrees=[0, 45, 92])

    folder = os.path.join(DATA_PATH, "samples", "rotation")
    if not os.path.exists(folder):
        os.makedirs(folder)

    for task_id, taskset in enumerate(scenario):
        taskset.plot(path=folder,
                     title="MNIST_Rotation_{}.jpg".format(task_id),
                     nb_samples=100,
                     shape=[28, 28, 1])


'''
Test the visualization with three tasks for permutations tasks
'''
@pytest.mark.slow
def test_visualization_permutations():
    scenario = Permutations(cl_dataset=MNIST(data_path=DATA_PATH, download=True, train=True),
                            nb_tasks=3,
                            seed=0)

    folder = os.path.join(DATA_PATH, "samples", "permutation")
    if not os.path.exists(folder):
        os.makedirs(folder)

    for task_id, taskset in enumerate(scenario):
        taskset.plot(path=folder,
                     title="MNIST_Permutations_{}.jpg".format(task_id),
                     nb_samples=100,
                     shape=[28, 28, 1])


'''
Test the visualization with three tasks for incremental tasks
'''
@pytest.mark.slow
def test_visualization_MNISTFellowship(DATA_PATH):
    cl_dataset = MNISTFellowship(data_path=DATA_PATH, download=True, train=True)
    scenario = ClassIncremental(cl_dataset=cl_dataset,
                                increment=10)

    folder = os.path.join(DATA_PATH, "samples", "fellowship")
    if not os.path.exists(folder):
        os.makedirs(folder)

    for task_id, taskset in enumerate(scenario):
        taskset.plot(path=folder,
                     title="MNISTFellowship_Incremental_{}.jpg".format(task_id),
                     nb_samples=100,
                     shape=[28, 28, 1])
