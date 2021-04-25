import numpy as np
import pytest
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import torch

from continuum.datasets import InMemoryDataset
from continuum.scenarios import ClassIncremental, InstanceIncremental, ContinualScenario


@pytest.fixture
def dataset():
    x = np.random.randint(0, 255, (100, 4, 4, 3), dtype=np.uint8)
    y = np.random.randint(0, 3, (100,), dtype=np.int16)
    t = np.ones_like(y)

    t[:30] = 0
    t[30:60] = 1
    t[60:] = 2

    return InMemoryDataset(x, y, t)



@pytest.mark.parametrize("scenario,opt", [
    (ClassIncremental, {'increment': 1}),
    (InstanceIncremental, {}),
    (ContinualScenario, {})
])
def test_same_transforms(dataset, scenario, opt):
    trsfs = [
        ToTensor(),
        lambda tensor: tensor.fill_(0)
    ]
    s = scenario(dataset, transformations=trsfs, **opt)

    for taskset in s:
        loader = DataLoader(taskset)
        for x, _, _ in loader:
            assert torch.unique(x).numpy().tolist() == [0]


@pytest.mark.parametrize("scenario,opt", [
    (ClassIncremental, {'increment': 1}),
    (InstanceIncremental, {}),
    (ContinualScenario, {})
])
def test_diff_transforms(dataset, scenario, opt):
    trsfs = [
        [ToTensor(), lambda tensor1: tensor1.fill_(0)],
        [ToTensor(), lambda tensor2: tensor2.fill_(1)],
        [ToTensor(), lambda tensor3: tensor3.fill_(2)],
    ]
    s = scenario(dataset, transformations=trsfs, **opt)

    for taskid, taskset in enumerate(s):
        loader = DataLoader(taskset)
        for x, _, _ in loader:
            assert torch.unique(x).numpy().tolist() == [taskid]


@pytest.mark.parametrize("scenario,opt,error", [
    (ClassIncremental, {'increment': 1}, True),
    (InstanceIncremental, {}, True),
    (ContinualScenario, {}, True),
    (ClassIncremental, {'increment': 1}, False),
    (InstanceIncremental, {}, False),
    (ContinualScenario, {}, False)
])
def test_diff_transforms_slice(dataset, scenario, opt, error):
    trsfs = [
        [ToTensor(), lambda tensor1: tensor1.fill_(0)],
        [ToTensor(), lambda tensor2: tensor2.fill_(1)],
        [ToTensor(), lambda tensor3: tensor3.fill_(2)],
    ]
    s = scenario(dataset, transformations=trsfs, **opt)

    for taskid in range(len(s)):
        if not error:
            taskset = s[taskid]
            loader = DataLoader(taskset)
            for x, _, _ in loader:
                assert torch.unique(x).numpy().tolist() == [taskid]
        else:
            with pytest.raises(ValueError):
                s[:taskid]





