import os

import pytest
import numpy as np
from torchvision.transforms import Resize, ToTensor

from continuum.datasets import CIFAR10
from continuum.scenarios import CIFAR2Spurious


DATA_PATH = os.environ.get("CONTINUUM_DATA_PATH")


@pytest.mark.slow
def test_spurious_nb_tasks():


    dataset = CIFAR10(DATA_PATH, train=True)
    NB_TASK = 5
    scenario_tr = CIFAR2Spurious(dataset, nb_tasks=NB_TASK, train=True)
    scenario_te = CIFAR2Spurious(dataset, nb_tasks=NB_TASK, train=False)

    assert scenario_tr.nb_tasks == NB_TASK
    assert scenario_te.nb_tasks == NB_TASK + 1
    assert scenario_te.nb_classes == scenario_tr.nb_classes == 2

@pytest.mark.slow
def test_spurious_support():
    # TODO
    pass

@pytest.mark.slow
def test_spurious_correlation():
    # TODO
    pass

'''
Test the visualization with instance scenario
'''
@pytest.mark.slow
def test_spurious_visualization():

    folder = "tests/samples/CIFAR2Spurious/"
    if not os.path.exists(folder):
        os.makedirs(folder)

    dataset = CIFAR10(DATA_PATH, train=True)
    NB_TASK = 5
    scenario_tr = CIFAR2Spurious(dataset, nb_tasks=NB_TASK, train=True)
    scenario_te = CIFAR2Spurious(dataset, nb_tasks=NB_TASK, train=False)

    for task_id, task_set_tr in enumerate(scenario_tr):
        task_set_tr.plot(path=folder,
                      title="CIFAR2Spurious_tr_{}.jpg".format(task_id),
                      nb_samples=100)
    for task_id, task_set_te in enumerate(scenario_te):
        task_set_te.plot(path=folder,
                      title="CIFAR2Spurious_te_{}.jpg".format(task_id),
                      nb_samples=100)
