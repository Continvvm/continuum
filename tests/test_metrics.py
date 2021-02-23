import numpy as np
import pytest
from torch import nn
import torch
from copy import deepcopy
from continuum.metrics import Logger

from continuum.metrics import get_model_size


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


def test_logger_nb_tasks(numpy_data):
    logger = Logger()
    all_targets, all_tasks = numpy_data
    nb_tasks = 3
    nb_epochs = 5
    for task in range(nb_tasks):
        for epoch in range(nb_epochs):
            for targets, task_ids in zip(all_targets, all_tasks):
                preds = np.copy(targets)
                logger.add([preds, targets, task_ids], subset="train")
            logger.end_epoch()
        logger.end_task()

    assert logger.nb_tasks == nb_tasks


def test_logger_simplest_add(numpy_data):
    logger = Logger()
    all_targets, all_tasks = numpy_data
    nb_tasks = 3
    nb_epochs = 5
    for task in range(nb_tasks):
        for epoch in range(nb_epochs):
            for targets, task_ids in zip(all_targets, all_tasks):
                preds = np.copy(targets)
                logger.add([preds, targets, task_ids], subset="train")
            logger.end_epoch()
        logger.end_task()


def test_logger_add_tensor(numpy_data):
    """
    test to check if we can use the logger to log random tensor with random keword
    """
    logger = Logger(list_keywords=['RandKeyword'])
    all_targets, all_tasks = numpy_data
    nb_tasks = 3
    nb_epochs = 5
    for task in range(nb_tasks):
        for epoch in range(nb_epochs):
            for targets, task_ids in zip(all_targets, all_tasks):
                rand_vector = torch.randn(15)
                logger.add(rand_vector, keyword='RandKeyword')
            logger.end_epoch()
        logger.end_task()


@pytest.mark.parametrize("mode,expected", [
    ("best", 1.), ("worst", 0.), ("random", None)
])
def test_metrics(numpy_data, mode, expected):
    logger = Logger()
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

        logger.add(value=[preds, targets, task_ids], subset="train")
        logger.add(value=[preds, targets, task_ids], subset="test")

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


@pytest.mark.parametrize("mode,expected", [
    ("best", 1.), ("worst", 0.), ("random", None)
])
def test_accuracy_per_task(numpy_data, mode, expected):
    logger = Logger()
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
        logger.add(value=[preds, targets, task_ids], subset="test")

    accuracies = logger.accuracy_per_task

    for accuracy in accuracies:
        assert 0. <= accuracy <= 1.0


@pytest.mark.parametrize("batch_size", [
    1, 32, None
])
def test_online_accuracy(numpy_data, batch_size):
    logger = Logger()
    all_targets, _ = numpy_data
    targets = all_targets[0]

    # we check that when no data is in the logger online_accuracy generate an error
    check_raised(lambda: logger.online_accuracy)

    batch_size = batch_size or len(targets)
    for batch_index in range(0, len(targets), batch_size):
        y = targets[batch_index: batch_index + batch_size]
        x = np.copy(y)

        logger.add(value=[x, y, None], subset="train")
        logger.online_accuracy
    logger.online_accuracy

    logger.add(value=[targets, np.copy(targets), None], subset="train")
    logger.online_accuracy


def test_require_subset_test(numpy_data):
    logger = Logger()
    check_raised(lambda: logger.accuracy)

    values = [numpy_data[0][0], numpy_data[0][0], numpy_data[0][1]]
    logger.add(values, subset="test")
    logger.accuracy


def test_require_subset_train(numpy_data):
    logger = Logger()
    check_raised(lambda: logger.online_cumulative_performance)

    values = [numpy_data[0][0], numpy_data[0][0], numpy_data[0][1]]
    logger.add(values, subset="train")
    logger.online_cumulative_performance


def test_model_size(torch_models):
    small, big = torch_models
    assert get_model_size(small) < get_model_size(big)


def test_model_growth(torch_models):
    small, big = torch_models

    logger1 = Logger(list_keywords=['model_size'])  # Logger declaration with parameter name
    logger1.add(get_model_size(small), keyword='model_size')
    logger1.end_task()
    logger1.add(get_model_size(small), keyword='model_size')
    logger1.end_task()
    ms1 = logger1.model_size_growth

    logger2 = Logger(['model_size'])  # Logger declaration without parameter name
    logger2.add(get_model_size(small), keyword='model_size')
    logger2.end_task()
    logger2.add(get_model_size(big), keyword='model_size')
    logger2.end_task()
    ms2 = logger2.model_size_growth

    logger3 = Logger(['model_size'])
    logger3.add(get_model_size(big), keyword='model_size')
    logger3.end_task()
    logger3.add(get_model_size(small), keyword='model_size')
    logger3.end_task()
    ms3 = logger3.model_size_growth

    logger4 = Logger(['model_size'])
    logger4.add(get_model_size(big), keyword='model_size')
    logger4.end_task()
    logger4.add(get_model_size(big), keyword='model_size')
    logger4.end_task()
    ms4 = logger4.model_size_growth

    assert ms1 == ms4 == ms3 == 1.0
    assert 0. <= ms2 < 1.


@pytest.mark.slow
def test_example_doc():
    from torch.utils.data import DataLoader
    import numpy as np

    from continuum import ClassIncremental
    from continuum.datasets import MNIST
    from continuum.metrics import Logger

    train_scenario = ClassIncremental(
        MNIST(data_path="my/data/path", download=True, train=True),
        increment=2
    )
    test_scenario = ClassIncremental(
        MNIST(data_path="my/data/path", download=True, train=False),
        increment=2
    )

    # model = ...

    logger = Logger(list_subsets=['train', 'test'])

    for task_id, (train_taskset, test_taskset) in enumerate(zip(train_scenario, test_scenario)):
        train_loader = DataLoader(train_taskset)
        test_loader = DataLoader(test_taskset)

        for x, y, t in train_loader:
            predictions = y  # model(x)

            logger.add([predictions, y, None], subset="train")
            _ = (f"Online accuracy: {logger.online_accuracy}")

        for x_test, y_test, t_test in test_loader:
            preds_test = y_test

            logger.add([preds_test, y_test, t_test], subset="test")

        _ = (f"Task: {task_id}, acc: {logger.accuracy}, avg acc: {logger.average_incremental_accuracy}")
        _ = (f"BWT: {logger.backward_transfer}, FWT: {logger.forward_transfer}")

        logger.end_task()


def check_raised(func):
    has_raised = False
    try:
        func()
    except:
        has_raised = True
    finally:
        assert has_raised
