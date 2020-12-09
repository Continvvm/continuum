import numpy as np
import pytest
from torch import nn
import torch

from continuum import MetricsLogger

# yapf: disable

@pytest.fixture
def numpy_data():
    nb_classes = 20
    nb_tasks = 5
    nb_data = 100
    seen_classes = 0
    inc_classes = nb_classes // nb_tasks

    targets = []
    tasks = []

    for t in range(nb_tasks):
        seen_classes += inc_classes

        task_targets = np.concatenate([
            np.ones(nb_data) * c
            for c in range(seen_classes)
        ])
        task_tasks = np.concatenate([
            np.ones(nb_data * inc_classes) * tt
            for tt in range(t + 1)
        ])  # We also see previous data

        targets.append(task_targets)
        tasks.append(task_tasks)

    return targets, tasks


@pytest.fixture
def torch_models():
    class Small(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Sequential(nn.Conv2d(2, 3, 1, bias=False), nn.Conv2d(2, 3, 1, bias=False))
            self.fc = nn.Linear(5, 4)
            self.scalar = nn.Parameter(torch.tensor(3.))

    class Big(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Sequential(nn.Conv2d(2, 3, 1, bias=False), nn.Conv2d(2, 3, 1, bias=False))
            self.fc = nn.Linear(5, 4)
            self.fc2 = nn.Linear(5, 4)
            self.scalar = nn.Parameter(torch.tensor(3.))

    return Small(), Big()


@pytest.mark.parametrize("mode,expected", [
    ("best", 1.), ("worst", 0.), ("random", None)
])
def test_metrics(numpy_data, mode, expected):
    logger = MetricsLogger()

    all_targets, all_tasks = numpy_data

    for targets, task_ids in zip(all_targets, all_tasks):
        if mode == "best":
            preds = np.copy(targets)
        elif mode == "worst":
            # Trick to never generate the correct predictions
            # only work for more three classes or more
            preds = (np.copy(targets) + 1) % np.max(targets)
        else:
            preds = np.random.randint(0, np.max(targets) + 1, targets.shape)

        logger.add(preds, targets, task_ids, subset="train")
        logger.add(preds, targets, task_ids, subset="test")

        accuracies = [
            logger.accuracy, logger.online_cumulative_performance,
            logger.accuracy_A,
        ]
        for acc in accuracies:
            if expected is not None:
                assert acc == expected, (acc, mode)
            assert 0. <= acc <= 1.
        assert -1. <= logger.backward_transfer <= 1.0
        assert 0. <= logger.positive_backward_transfer <= 1.0
        assert 0. <= logger.remembering <= 1.0
        assert 0. <= logger.forgetting <= 1.0



    assert 0. <= logger.average_incremental_accuracy <= 1.


def test_require_subset_test(numpy_data):
    logger = MetricsLogger()

    has_raised = False
    try:
        logger.accuracy
    except:
        has_raised = True
    finally:
        assert has_raised

    logger.add(numpy_data[0][0], numpy_data[0][0], numpy_data[0][1], subset="test")
    logger.accuracy


def test_require_subset_train(numpy_data):
    logger = MetricsLogger()

    has_raised = False
    try:
        logger.online_cumulative_performance
    except:
        has_raised = True
    finally:
        assert has_raised

    logger.add(numpy_data[0][0], numpy_data[0][0], numpy_data[0][1], subset="train")
    logger.online_cumulative_performance


def test_model_efficiency(torch_models):
    small, big = torch_models

    logger1 = MetricsLogger()
    logger1.add(model=small)
    logger1.add(model=small)
    ms1 = logger1.model_size_efficiency

    logger2 = MetricsLogger()
    logger2.add(model=small)
    logger2.add(model=big)
    ms2 = logger2.model_size_efficiency

    logger3 = MetricsLogger()
    logger3.add(model=big)
    logger3.add(model=small)
    ms3 = logger3.model_size_efficiency

    logger4 = MetricsLogger()
    logger4.add(model=big)
    logger4.add(model=big)
    ms4 = logger4.model_size_efficiency

    assert ms1 == ms4 == ms3 == 1.0
    assert 0. <= ms2 < 1.
