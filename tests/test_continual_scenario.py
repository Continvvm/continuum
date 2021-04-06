import numpy as np
import pytest

from continuum.datasets import InMemoryDataset
from continuum.scenarios import ContinualScenario

# yapf: disable

def test_scenario():
    x = np.ones((100, 4, 4, 3), dtype=np.uint8)
    y = np.arange(100) // 5

    nb_tasks = 10
    t = np.random.randint(nb_tasks, size=100)

    dummy = InMemoryDataset(x, y, t)
    scenario = ContinualScenario(dummy)

    assert scenario.nb_tasks == nb_tasks


def test_bad_task_ids():
    x = np.ones((100, 4, 4, 3), dtype=np.uint8)
    y = np.arange(100) // 5
    nb_tasks = 10
    # test if one missing generate an error
    t = np.random.randint(10, size=100)
    t = t + np.ones(100)  # shift indexes from [0 - 9] to [1 - 10]

    dummy = InMemoryDataset(x, y, t)

    with pytest.raises(Exception):
        scenario = ContinualScenario(dummy)
