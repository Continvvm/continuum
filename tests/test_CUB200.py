import os

import pytest
from torchvision.transforms import Resize, ToTensor
from torch.utils.data import DataLoader

from continuum.datasets import CUB200
from continuum.scenarios import ClassIncremental

DATA_PATH = os.environ.get("CONTINUUM_DATA_PATH")

'''
Test the visualization with instance_class scenario
'''
@pytest.mark.slow
def test_scenario_CUB200_ClassIncremental():
    dataset = CUB200(DATA_PATH, train=True)
    scenario = ClassIncremental(dataset, increment=100, transformations=[Resize((224, 224)), ToTensor()])

    print(f"Nb classes : {scenario.nb_classes} ")
    print(f"Nb tasks : {scenario.nb_tasks} ")
    for task_id, task_set in enumerate(scenario):
        print(f"Task {task_id} : {task_set.nb_classes} classes")
        task_set.plot(path="Archives/Samples/CUB200/CI",
                      title="CUB200_ClassIncremental_{}.jpg".format(task_id),
                      nb_samples=100)

@pytest.mark.slow
def test_train_test_CUB200():
    dataset_tr = CUB200(DATA_PATH, train=True)
    dataset_te = CUB200(DATA_PATH, train=False)

    scenario_tr = ClassIncremental(dataset_tr, nb_tasks=1)
    scenario_te = ClassIncremental(dataset_te, nb_tasks=1)

    assert len(scenario_tr[0]) != len(scenario_te[0])

    for taskset in scenario_tr:
        loader = DataLoader(taskset)
        _, _, _ = next(iter(loader))

    for taskset in scenario_te:
        loader = DataLoader(taskset)
        _, _, _ = next(iter(loader))
