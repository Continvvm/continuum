import os

import pytest
from torchvision.transforms import Resize, ToTensor

from continuum.datasets import CIFAR100
from continuum.scenarios import ClassIncremental, ContinualScenario

DATA_PATH = os.environ.get("CONTINUUM_DATA_PATH")

'''
Basic test of CIFAR100 without parameters
'''
@pytest.mark.slow
def test_scenario_CIFAR100_ClassIncremental():
    dataset = CIFAR100(DATA_PATH, train=True)
    scenario = ClassIncremental(dataset, increment=50)

    assert scenario.nb_classes == 100
    assert scenario.nb_tasks == 2

'''
Basic test of CIFAR100 without parameters
'''
@pytest.mark.slow
def test_scenario_CIFAR100_CoarseLabels():
    dataset = CIFAR100(DATA_PATH, train=True, labels_type="category")
    scenario = ClassIncremental(dataset, increment=10)

    assert scenario.nb_classes == 20
    assert scenario.nb_tasks == 2

'''
Basic test of CIFAR100 without parameters
'''
@pytest.mark.slow
def test_scenario_CIFAR100_Scenarios():
    dataset = CIFAR100(DATA_PATH, train=True, labels_type="category", task_labels="category")
    scenario = ContinualScenario(dataset)
    assert scenario.nb_classes == 20
    assert scenario.nb_tasks == 20

    dataset = CIFAR100(DATA_PATH, train=True, labels_type="category", task_labels="class")
    scenario = ContinualScenario(dataset)
    assert scenario.nb_classes == 20
    assert scenario.nb_tasks == 100

    dataset = CIFAR100(DATA_PATH, train=True, labels_type="class", task_labels="class")
    scenario = ContinualScenario(dataset)
    assert scenario.nb_classes == 100
    assert scenario.nb_tasks == 100

    dataset = CIFAR100(DATA_PATH, train=True, labels_type="class", task_labels="category")
    scenario = ContinualScenario(dataset)
    assert scenario.nb_classes == 100
    assert scenario.nb_tasks == 20


    dataset = CIFAR100(DATA_PATH, train=True, labels_type="category", task_labels="lifelong")
    scenario = ContinualScenario(dataset)
    assert scenario.nb_classes == 20
    assert scenario.nb_tasks == 5
