import numpy as np
import pytest

from continuum.datasets import InMemoryDataset
from continuum.scenarios import ALMA

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

@pytest.mark.parametrize("nb_tasks,nb_tasks_gt", [
    (2, 2),
    (6, 6),
])
def test_instance_default_nb_tasks(numpy_data_per_task, nb_tasks, nb_tasks_gt):
    """Test the ALMA loader when the dataset does provide
    a default number of tasks."""
    train, test = numpy_data_per_task

    x_train, y_train, t_train = train

    dummy = InMemoryDataset(x_train, y_train, t=t_train)

    scenario = ALMA(dummy, nb_megabatches=nb_tasks)
    nb_classes = scenario.nb_classes

    assert len(scenario) == nb_tasks_gt, (nb_tasks, nb_tasks_gt)
    for task_id, train_dataset in enumerate(scenario):
        assert nb_classes == len(np.unique(train_dataset._y))

        unique_pixels = np.unique(train_dataset._x)
        if nb_tasks is None:
            assert len(unique_pixels) == 1 and unique_pixels[0] == float(task_id)


@pytest.mark.parametrize("nb_tasks,error", [(2000, "error"), (300, None)])
def test_too_many_tasks(numpy_data_per_task, nb_tasks, error):
    train, test = numpy_data_per_task
    x_train, y_train, t_train = train

    dummy = InMemoryDataset(x_train, y_train, t=t_train)

    if error == "error":
        with pytest.raises(Exception):
            scenario = ALMA(dummy, nb_megabatches=nb_tasks)
    elif error == "warning":
        with pytest.warns(Warning):
            scenario = ALMA(dummy, nb_megabatches=nb_tasks)
    else:
        scenario = ALMA(dummy, nb_megabatches=nb_tasks)



@pytest.mark.parametrize("nb_tasks", [None, 0, -1])
def test_invalid(numpy_data_per_task, nb_tasks):
    train, test = numpy_data_per_task

    x_train, y_train, t_train = train
    x_test, y_test, t_test = test

    dummy = InMemoryDataset(x_train, y_train)

    with pytest.raises(Exception):
        ALMA(dummy, nb_megabatches=nb_tasks)


@pytest.fixture
def equal_data():
    x = np.ones((100, 4, 4, 3), dtype=np.uint8)
    y = np.concatenate((
        np.ones((25,), dtype=np.uint32) * 0,
        np.ones((50,), dtype=np.uint32) * 1,
        np.ones((10,), dtype=np.uint32) * 2,
        np.ones((15,), dtype=np.uint32) * 3,
    ))
    return x, y


@pytest.fixture
def unequal_data():
    y = np.concatenate((
        np.ones((22,), dtype=np.uint32) * 0,
        np.ones((53,), dtype=np.uint32) * 1,
        np.ones((19,), dtype=np.uint32) * 2,
        np.ones((7,), dtype=np.uint32) * 3,
    ))
    x = np.ones((len(y), 4, 4, 3), dtype=np.uint8)
    return x, y


@pytest.mark.parametrize("nb_tasks,nb_tasks_gt", [
    (2, 2),
    (6, 6),
])

def test_data_is_shuffled(numpy_data_per_task, nb_tasks, nb_tasks_gt):
    """Make sure the data is shuffled before splitting into tasks"""
    train, test = numpy_data_per_task

    x_train, y_train, t_train = train
    items_per_task = x_train.shape[0] // nb_tasks

    dummy = InMemoryDataset(x_train, y_train, t=t_train)

    scenario = ALMA(dummy, nb_megabatches=nb_tasks)
    nb_classes = scenario.nb_classes

    assert len(scenario) == nb_tasks_gt, (nb_tasks, nb_tasks_gt)
    for task_id, train_dataset in enumerate(scenario):
        assert nb_classes == len(np.unique(train_dataset._y))

        task_x = train_dataset._x
        unshuffled_task_x = x_train[task_id*items_per_task:(task_id+1)*items_per_task]

        assert task_x.shape == unshuffled_task_x.shape
        assert not np.allclose(task_x, unshuffled_task_x)

