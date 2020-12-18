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

        logger.add_step(preds, targets, task_ids, subset="train")
        logger.add_step(preds, targets, task_ids, subset="test")

        accuracies = [
            logger.accuracy, logger.online_cumulative_performance,
            logger.accuracy_A,
        ]
        for acc in accuracies:
            if expected is not None:
                assert acc == expected, (acc, mode)
            assert 0. <= acc <= 1.
        assert -1. <= logger.backward_transfer <= 1.0
        assert -1. <= logger.forward_transfer <= 1.0
        assert 0. <= logger.positive_backward_transfer <= 1.0
        assert 0. <= logger.remembering <= 1.0
        assert 0. <= logger.forgetting <= 1.0

    assert 0. <= logger.average_incremental_accuracy <= 1.


@pytest.mark.parametrize("batch_size", [
    1, 32, None
])
def test_online_accuracy(numpy_data, batch_size):
    logger = MetricsLogger()
    all_targets, _ = numpy_data
    targets = all_targets[0]

    check_raised(lambda: logger.online_accuracy)

    batch_size = batch_size or len(targets)
    for batch_index in range(0, len(targets), batch_size):
        y = targets[batch_index: batch_index+batch_size]
        x = np.copy(y)

        logger.add_batch(x, y)
        logger.online_accuracy
    logger.online_accuracy

    logger.add_step(targets, np.copy(targets))
    check_raised(lambda: logger.online_accuracy)


def test_require_subset_test(numpy_data):
    logger = MetricsLogger()
    check_raised(lambda: logger.accuracy)

    logger.add_step(numpy_data[0][0], numpy_data[0][0], numpy_data[0][1], subset="test")
    logger.accuracy


def test_require_subset_train(numpy_data):
    logger = MetricsLogger()
    check_raised(lambda: logger.online_cumulative_performance)

    logger.add_step(numpy_data[0][0], numpy_data[0][0], numpy_data[0][1], subset="train")
    logger.online_cumulative_performance


def test_model_efficiency(torch_models):
    small, big = torch_models

    logger1 = MetricsLogger()
    logger1.add_step(model=small)
    logger1.add_step(model=small)
    ms1 = logger1.model_size_efficiency

    logger2 = MetricsLogger()
    logger2.add_step(model=small)
    logger2.add_step(model=big)
    ms2 = logger2.model_size_efficiency

    logger3 = MetricsLogger()
    logger3.add_step(model=big)
    logger3.add_step(model=small)
    ms3 = logger3.model_size_efficiency

    logger4 = MetricsLogger()
    logger4.add_step(model=big)
    logger4.add_step(model=big)
    ms4 = logger4.model_size_efficiency

    assert ms1 == ms4 == ms3 == 1.0
    assert 0. <= ms2 < 1.


@pytest.mark.slow
def test_example_doc():
    from torch.utils.data import DataLoader
    import numpy as np

    from continuum import MetricsLogger, ClassIncremental
    from continuum.datasets import MNIST

    train_scenario = ClassIncremental(
        MNIST(data_path="/tmp", download=True, train=True),
        increment=2
     )
    test_scenario = ClassIncremental(
        MNIST(data_path="/tmp", download=True, train=False),
        increment=2
     )

    logger = MetricsLogger()

    for task_id, (train_taskset, test_taskset) in enumerate(zip(train_scenario, test_scenario)):
        train_loader = DataLoader(train_taskset)
        test_loader = DataLoader(test_taskset)

        for x, y, t in train_loader:
            predictions = torch.clone(y)

            logger.add_batch(predictions, y)
            _ = (f"Online accuracy: {logger.online_accuracy}")

        preds, targets, task_ids = [], [], []
        for x, y, t in test_loader:
            preds.append(y.cpu().numpy())
            targets.append(y.cpu().numpy())
            task_ids.append(t.cpu().numpy())

        logger.add_step(
            np.concatenate(preds),
            np.concatenate(targets),
            np.concatenate(task_ids)
        )
        _ = (f"Task: {task_id}, acc: {logger.accuracy}, avg acc: {logger.average_incremental_accuracy}")
        _ = (f"BWT: {logger.backward_transfer}, FWT: {logger.forward_transfer}")



def check_raised(func):
    has_raised = False
    try:
        func()
    except:
        has_raised = True
    finally:
        assert has_raised


