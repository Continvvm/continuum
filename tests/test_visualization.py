import numpy as np
import pytest
import os
from torchvision import transforms
from torch.utils.data import DataLoader

from continuum.datasets import MNIST, CIFAR10, CIFAR100, KMNIST, FashionMNIST, TinyImageNet200, AwA2, Core50
from continuum.scenarios import Rotations
from continuum.scenarios import Permutations
from continuum.scenarios import ClassIncremental
from continuum.datasets import MNISTFellowship

DATA_PATH = os.environ.get("CONTINUUM_DATA_PATH")


@pytest.mark.slow
@pytest.mark.parametrize("dataset, name, shape, init_increment, increment", [(MNIST, "MNIST", [28, 28, 1], 0, 2),
                                                                             (KMNIST, "KMNIST", [28, 28, 1], 0, 2),
                                                                             (FashionMNIST, "FashionMNIST", [28, 28, 1],
                                                                              0, 2),
                                                                             (CIFAR10, "CIFAR10", [32, 32, 3], 0, 2),
                                                                             (CIFAR100, "CIFAR100", [32, 32, 3], 0, 20),
                                                                             (AwA2, "AwA2", [224, 224, 3], 7, 5),
                                                                             (Core50, "Core50", [224, 224, 3], 0, 10),
                                                                             (TinyImageNet200, "TinyImageNet200",
                                                                              [64, 64, 3], 0, 40)])
def test_visualization_ClassIncremental(dataset, name, shape, init_increment, increment):
    trsf = None
    if name == "AwA2":
        trsf = [transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])]
    elif name == "Core50":
        trsf = [transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])]
    scenario = ClassIncremental(cl_dataset=dataset(data_path=DATA_PATH, download=True, train=True),
                                increment=increment, initial_increment=init_increment, transformations=trsf)

    folder = "tests/samples/class_incremental/"
    if not os.path.exists(folder):
        os.makedirs(folder)

    for task_id, taskset in enumerate(scenario):
        taskset.plot(path=folder,
                     title="{}_ClassIncremental_{}.jpg".format(name, task_id),
                     nb_samples=100,
                     shape=shape)
        loader = DataLoader(taskset)
        _, _, _ = next(iter(loader))


'''
Test the visualization with three tasks for rotations tasks
'''


@pytest.mark.slow
def test_visualization_rotations():
    scenario = Rotations(cl_dataset=MNIST(data_path=DATA_PATH, download=True, train=True),
                         nb_tasks=3,
                         list_degrees=[0, 45, 92])

    folder = "tests/samples/rotations/"
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

    folder = "tests/samples/permutations/"
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
def test_visualization_MNISTFellowship():
    cl_dataset = MNISTFellowship(data_path=DATA_PATH, download=True, train=True)
    scenario = ClassIncremental(cl_dataset=cl_dataset,
                                increment=10)

    folder = "tests/samples/fellowship/"
    if not os.path.exists(folder):
        os.makedirs(folder)

    for task_id, taskset in enumerate(scenario):
        taskset.plot(path=folder,
                     title="MNISTFellowship_Incremental_{}.jpg".format(task_id),
                     nb_samples=100,
                     shape=[28, 28, 1])


'''
Test MNIST 360 Scenario
'''
import torch
from continuum.viz import visualize_batch


@pytest.mark.slow
def test_visualization_MNIST360():
    folder = "tests/samples/mnist360/"
    if not os.path.exists(folder):
        os.makedirs(folder)
    continual_dataset = MNIST(data_path=DATA_PATH, download=False, train=True)

    for i in range(3):
        start_angle = 120 * i
        angle = int(120 / 9)

        list_rotation_first = [[transforms.RandomAffine(degrees=[start_angle + angle * j, start_angle + angle * j + 5])]
                               for j in range(10)]
        list_rotation_second = [
            [transforms.RandomAffine(degrees=[start_angle + 45 + angle * j, start_angle + 45 + angle * j + 5])] for j in
            range(10)]
        first_digit_scenario = ClassIncremental(continual_dataset, increment=1, transformations=list_rotation_first)
        second_digit_scenario = ClassIncremental(continual_dataset, increment=1, transformations=list_rotation_second)

        for task_id in range(9):
            taskset_1 = first_digit_scenario[task_id]
            taskset_2 = second_digit_scenario[1 + task_id % 9]

            # / ! \ we can not concatenate taskset here, since transformations would not work correctly

            loader_1 = DataLoader(taskset_1, batch_size=64)
            loader_2 = DataLoader(taskset_2, batch_size=64)

            nb_minibatches = min(len(loader_1), len(loader_2))
            for minibatch in range(nb_minibatches):
                x_1, y_1, t_1 = next(iter(loader_1))
                x_2, y_2, t_2 = next(iter(loader_2))

                x, y, t = torch.cat([x_1, x_2]), torch.cat([y_1, y_2]), torch.cat([t_1, t_2])

                # train here on x, y, t

                #### to visualize result ####
                # visualize_batch(batch=x[:100], number=100, shape=[28, 28, 1], path=folder + f"MNIST360_{task_id + 9 * i}.jpg")
