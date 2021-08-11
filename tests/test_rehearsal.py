import pytest
import torch
import numpy as np

from continuum.datasets import InMemoryDataset
from continuum import ClassIncremental
from continuum import rehearsal


@pytest.fixture
def scenario():
    x = np.random.randn(100, 2)
    y = np.concatenate([
        np.ones(10) * i
        for i in range(10)
    ])
    t = None

    dataset = InMemoryDataset(x, y, t)
    return ClassIncremental(dataset, increment=2)


@pytest.mark.parametrize("nb,method", [
    (10, rehearsal.herd_random),
    (20, rehearsal.herd_random),
    (5, rehearsal.herd_random),
    (10, rehearsal.herd_closest_to_barycenter),
    (20, rehearsal.herd_closest_to_barycenter),
    (5, rehearsal.herd_closest_to_barycenter),
    (10, rehearsal.herd_closest_to_cluster),
    (20, rehearsal.herd_closest_to_cluster),
    (5, rehearsal.herd_closest_to_cluster),
])
def test_nb_sampled(scenario, nb, method):
    for taskset in scenario:
        x, y, t = taskset.get_raw_samples()

        mx, my, mt = method(x, y, t, x, nb)
        assert len(mx) == 2 * min(nb, 10)



@pytest.mark.parametrize("memory_size,method,fixed", [
    (200, rehearsal.herd_random, True),
    (100, rehearsal.herd_random, True),
    (50, rehearsal.herd_random, True),
    (200, rehearsal.herd_random, False),
    (100, rehearsal.herd_random, False),
    (50, rehearsal.herd_random, False),
])
def test_memory(scenario, memory_size, method, fixed):
    memory = rehearsal.RehearsalMemory(
        memory_size, method, fixed, 10
    )
    assert len(memory) == 0

    c = 0
    for taskset in scenario:
        x, y, t = taskset.get_raw_samples()

        c += 2
        memory.add(x, y, t, x)
        assert memory.nb_classes == c
        if fixed:
            nb_per_class = min(memory_size, 100) // 10
            assert len(memory) == nb_per_class * c
        else:
            ideal_size = min(min(memory_size, 100), 10 * c)
            assert ideal_size >= len(memory) >= ideal_size - 5



@pytest.mark.parametrize("name,method", [
    ("random", rehearsal.herd_random),
    ("barycenter", rehearsal.herd_closest_to_barycenter),
    ("cluster", rehearsal.herd_closest_to_cluster),
])
def test_memory_name(name, method):
    memory = rehearsal.RehearsalMemory(
        20, name, True, 10
    )
    assert memory.herding_method == method

