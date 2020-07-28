import numpy as np
import pytest
import os

from continuum.datasets import MNIST
from continuum.scenarios import Rotations
from continuum.scenarios import Permutations
from continuum.scenarios import ClassIncremental

'''
Test the visualization with three tasks for rotations tasks
'''
@pytest.mark.slow
def test_visualization_rotations():
    clloader = Rotations(cl_dataset=MNIST("./pytest/Samples/Datasets", download=True, train=True),
                         nb_tasks=3,
                         list_degrees=[0, 45, 92])

    folder = "./tests/Samples/Rotations/"
    if not os.path.exists(folder):
        os.makedirs(folder)

    for task_id, train_dataset in enumerate(clloader):
        train_dataset.plot(path=folder,
                           title="MNIST_Rotation_{}.jpg".format(task_id),
                           nb_samples=100,
                           shape=[28, 28, 1])


'''
Test the visualization with three tasks for permutations tasks
'''
@pytest.mark.slow
def test_visualization_permutations():
    clloader = Permutations(cl_dataset=MNIST("./pytest/Samples/Datasets", download=True, train=True),
                            nb_tasks=3,
                            seed=0)

    folder = "./tests/Samples/Permutations/"
    if not os.path.exists(folder):
        os.makedirs(folder)

    for task_id, train_dataset in enumerate(clloader):
        train_dataset.plot(path=folder,
                           title="MNIST_Permutations_{}.jpg".format(task_id),
                           nb_samples=100,
                           shape=[28, 28, 1])

'''
Test the visualization with three tasks for incremental tasks
'''
@pytest.mark.slow
def test_visualization_incremental():
    clloader = ClassIncremental(cl_dataset=MNIST("./pytest/Samples/Datasets", download=True, train=True),
                                nb_tasks=5,
                                increment=2)

    folder = "./tests/Samples/Incremental/"
    if not os.path.exists(folder):
        os.makedirs(folder)

    for task_id, train_dataset in enumerate(clloader):
        train_dataset.plot(path=folder,
                           title="MNIST_Incremental_{}.jpg".format(task_id),
                           nb_samples=100,
                           shape=[28, 28, 1])
