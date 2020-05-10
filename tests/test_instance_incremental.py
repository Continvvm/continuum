import numpy as np
import pytest

from continuum.datasets import InMemoryDataset
from continuum.scenarios import InstanceIncremental

# yapf: disable


@pytest.fixture
def numpy_data():
    nb_classes = 6
    nb_data = 100

    x_train = []
    y_train = []
    for i in range(nb_classes):
        x_train.append(np.ones((nb_data, 4, 4, 3), dtype=np.uint8) * i)
        y_train.append(np.ones(nb_data) * i)
    x_train = np.concatenate(x_train)
    y_train = np.concatenate(y_train)

    x_test = np.copy(x_train)
    y_test = np.copy(y_train)

    return (x_train, y_train.astype(int)), (x_test, y_test.astype(int))


@pytest.mark.parametrize("nb_tasks,nb_tasks_gt", [
    (2, 2),
    (6, 6),
    (1, 1),
])
def test_instance_auto_nb_tasks(numpy_data, nb_tasks, nb_tasks_gt):
    """Test the InstanceIncremental loader when the dataset doesn't provide
    any default number of tasks."""
    train, test = numpy_data
    dummy = InMemoryDataset(*train)
    clloader = InstanceIncremental(dummy, nb_tasks=nb_tasks)

    nb_classes = clloader.nb_classes

    assert len(clloader) == nb_tasks_gt
    for task_id, train_dataset in enumerate(clloader):
        assert nb_classes == len(np.unique(train_dataset._y))


@pytest.fixture
def numpy_data_per_task():
    nb_classes = 6
    nb_tasks = 3
    nb_data = 100

    x_train = []
    y_train = []
    t_train = []
    for i in range(nb_tasks):
        for j in range(nb_classes):
            x_train.append(np.ones((nb_data, 4, 4, 3), dtype=np.uint8) * i)
            y_train.append(np.ones(nb_data) * j)
            t_train.append(np.ones(nb_data) * i)
    x_train = np.concatenate(x_train)
    y_train = np.concatenate(y_train)
    t_train = np.concatenate(t_train)

    x_test = np.copy(x_train)
    y_test = np.copy(y_train)
    t_test = np.copy(t_train)

    return (x_train, y_train.astype(int), t_train), (x_test, y_test.astype(int), t_test)


@pytest.mark.parametrize("nb_tasks,nb_tasks_gt,catch", [
    (2, 2, False),
    (6, None, True),
    (0, 3, False),
])
def test_instance_default_nb_tasks(numpy_data_per_task, nb_tasks, nb_tasks_gt, catch):
    """Test the InstanceIncremental loader when the dataset does provide
    a default number of tasks."""
    train, test = numpy_data_per_task

    x_train, y_train, t_train = train
    x_test, y_test, t_test = test

    dummy = InMemoryDataset(x_train, y_train, t_=t_train)

    has_raised = False
    try:
        clloader = InstanceIncremental(dummy, nb_tasks=nb_tasks)
    except Exception:
        has_raised = True

    if catch:
        assert has_raised
        return
    else:
        assert not has_raised

    nb_classes = clloader.nb_classes

    assert len(clloader) == nb_tasks_gt
    for task_id, train_dataset in enumerate(clloader):
        assert nb_classes == len(np.unique(train_dataset._y))

        unique_pixels = np.unique(train_dataset._x)
        assert len(unique_pixels) == 1 and unique_pixels[0] == float(task_id)
