import numpy as np
import pytest
from torch.utils.data import DataLoader
from continuum import scenarios

from continuum.scenarios import ClassIncremental
from continuum.datasets import MNIST, CIFAR10, CIFAR100, KMNIST, FashionMNIST, InMemoryDataset
from torchvision.transforms import transforms


# yapf: disable

@pytest.fixture
def dataset():
    x = np.zeros((100, 200))
    y = np.random.randint(0, 100, (100, 4))
    t = np.concatenate([
        np.ones((25,)) * 0,
        np.ones((25,)) * 1,
        np.ones((25,)) * 2,
        np.ones((25,)) * 3,
    ])
    y[:25, 0] = 0
    y[25:50, 0] = 1
    y[50:75, 0] = 2
    y[75:, 0] = 3

    return InMemoryDataset(x, y, t)


def test_multitarget(dataset):
    scenario = ClassIncremental(dataset, increment=1)
    assert len(scenario) == 4
    assert (scenario.class_order == np.array([0, 1, 2, 3])).all()

    for task_id, taskset in enumerate(scenario):
        loader = DataLoader(taskset, batch_size=100)
        _, y, _ = next(iter(loader))

        assert len(y.shape) == 2
        u = np.unique(y[:, 0])
        assert len(u) == 1 and u[0] == task_id

