import numpy as np
import pytest
from continuum.scenarios import Rotations
from tests.test_classorder import InMemoryDatasetTest
from continuum.datasets import MNIST, CIFAR100


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


'''
Test the initialization with three tasks
'''


def test_init(numpy_data):
    train, test = numpy_data
    dummy = InMemoryDatasetTest(*train)

    Trsf_0 = 0
    Trsf_1 = (15, 20)
    Trsf_2 = 45

    list_degrees = [Trsf_0, Trsf_1, Trsf_2]

    scenario = Rotations(cl_dataset=dummy, nb_tasks=3, list_degrees=list_degrees)

    for task_id, train_dataset in enumerate(scenario):
        continue


@pytest.mark.parametrize("shared_label_space", [True, False])
def test_shared_labels(numpy_data, shared_label_space):
    train, test = numpy_data
    dummy = InMemoryDatasetTest(*train)
    list_degrees = [0, 15, 45]

    scenario = Rotations(cl_dataset=dummy, nb_tasks=3, list_degrees=list_degrees, shared_label_space=shared_label_space)

    previous_classes = None
    for task_id, taskset in enumerate(scenario):
        classes = taskset.get_classes()
        if task_id > 0:
            if shared_label_space:
                assert (classes == previous_classes).all()
            else:
                assert (classes == previous_classes + len(classes)).all()

        previous_classes = classes


def test_fail_init(numpy_data):
    train, test = numpy_data
    dummy = InMemoryDatasetTest(*train)

    Trsf_0 = 2
    Trsf_1 = (15, 20, 25)  # non sens
    Trsf_2 = 45

    list_degrees = [Trsf_0, Trsf_1, Trsf_2]

    # should detect that a transformation is non-sens in the list
    with pytest.raises(ValueError):
        Rotations(cl_dataset=dummy, nb_tasks=3, list_degrees=list_degrees)


@pytest.mark.slow
@pytest.mark.parametrize("shared_label_space", [True, False])
@pytest.mark.parametrize("dataset", [MNIST, CIFAR100])
def test_with_dataset(dataset, shared_label_space):
    dataset = dataset(data_path="./tests/Datasets", download=True, train=True)
    list_degrees = [0, 45, 90]
    scenario = Rotations(cl_dataset=dataset,
                          nb_tasks=3,
                          list_degrees=list_degrees,
                          shared_label_space=shared_label_space)

    for task_id, taskset in enumerate(scenario):

        classes = taskset.get_classes()

        if shared_label_space:
            assert len(classes) == classes.max() + 1
        else:
            assert len(classes) == classes.max() + 1 - (task_id * len(classes))
