import numpy as np
import pytest
from torch.utils.data import DataLoader

from continuum.datasets import PermutedMNIST, RotatedMNIST
from continuum.scenarios import ClassIncremental

# yapf: disable


@pytest.mark.parametrize("nb_permutations,nb_tasks", [
    (0, 1),
    (4, 5)
])
def test_permuted(tmp_path_factory, nb_permutations, nb_tasks):
    dataset = PermutedMNIST(
        data_path=tmp_path_factory.getbasetemp(),
        download=True,
        nb_permutations=nb_permutations
    )

    clloader = ClassIncremental(dataset, nb_tasks=nb_tasks, increment=10)

    assert clloader.nb_tasks == nb_tasks
    seen_tasks = 0
    for task_id, train_dataset in enumerate(clloader):
        seen_tasks += 1

        for _ in DataLoader(train_dataset):
            pass

        assert np.max(train_dataset.y) == 9
        assert np.min(train_dataset.y) == 0
    assert seen_tasks == nb_tasks


@pytest.mark.parametrize("angles,nb_tasks", [
    ([], 1),
    ([45, 180], 3),
    ([45, 90, 135, 180], 5)
])
def test_rotated(tmp_path_factory, angles, nb_tasks):
    dataset = RotatedMNIST(
        data_path=tmp_path_factory.getbasetemp(),
        download=True,
        angles=angles
    )

    clloader = ClassIncremental(dataset, nb_tasks, increment=10)

    assert clloader.nb_tasks == nb_tasks
    seen_tasks = 0
    for task_id, train_dataset in enumerate(clloader):
        seen_tasks += 1

        for _ in DataLoader(train_dataset):
            pass

        assert np.max(train_dataset.y) == 9
        assert np.min(train_dataset.y) == 0
    assert seen_tasks == nb_tasks
