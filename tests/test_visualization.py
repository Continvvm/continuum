import numpy as np
import pytest
import os

from continuum.datasets import MNIST
from continuum.scenarios import Rotations
from continuum.scenarios import Permutations
from continuum.scenarios import ClassIncremental
from continuum.datasets import MNISTFellowship

'''
Test the visualization with three tasks for rotations tasks
'''
@pytest.mark.slow
def test_visualization_rotations():
    scenario = Rotations(cl_dataset=MNIST(data_path="./pytest/Samples/Datasets", download=True, train=True),
                         nb_tasks=3,
                         list_degrees=[0, 45, 92])

    folder = "./tests/Samples/Rotations/"
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
    scenario = Permutations(cl_dataset=MNIST(data_path="./pytest/Samples/Datasets", download=True, train=True),
                            nb_tasks=3,
                            seed=0)

    folder = "./tests/Samples/Permutations/"
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
def test_visualization_incremental():
    scenario = ClassIncremental(cl_dataset=MNIST(data_path="./pytest/Samples/Datasets", download=True, train=True),
                                nb_tasks=5,
                                increment=2)

    folder = "./tests/Samples/Incremental/"
    if not os.path.exists(folder):
        os.makedirs(folder)

    for task_id, taskset in enumerate(scenario):
        taskset.plot(path=folder,
                           title="MNIST_Incremental_{}.jpg".format(task_id),
                           nb_samples=100,
                           shape=[28, 28, 1])

'''
Test the visualization with three tasks for incremental tasks
'''
@pytest.mark.slow
def test_visualization_MNISTFellowship():

    cl_dataset = MNISTFellowship(data_path="./pytest/Samples/Datasets", download=True, train=True)
    scenario = ClassIncremental(cl_dataset=cl_dataset,
                                increment=10)

    folder = "./tests/Samples/MNISTFellowship/"
    if not os.path.exists(folder):
        os.makedirs(folder)

    for task_id, taskset in enumerate(scenario):
        taskset.plot(path=folder,
                           title="MNISTFellowship_Incremental_{}.jpg".format(task_id),
                           nb_samples=100,
                           shape=[28, 28, 1])