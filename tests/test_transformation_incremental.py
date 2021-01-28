import numpy as np
import pytest
import torch
from PIL import Image
from torchvision.transforms import transforms

from continuum.datasets import InMemoryDataset
from continuum.scenarios import TransformationIncremental

NB_CLASSES = 6


@pytest.fixture
def numpy_data():
    nb_data = 100  # not too small to have all classes

    x_train = []
    y_train = []
    x_train.append(
        np.array([np.random.randint(100, size=(2, 2, 3)).astype(dtype=np.uint8)] * nb_data)
    )
    y_train.append(np.random.randint(NB_CLASSES, size=(nb_data)))
    x_train = np.concatenate(x_train)
    y_train = np.concatenate(y_train)

    return x_train, y_train.astype(int)


'''
Test the initialization with three tasks
'''


def test_init(numpy_data):
    x, y = numpy_data
    dummy = InMemoryDataset(x, y, train='train')

    Trsf_0 = []
    Trsf_1 = [transforms.RandomAffine(degrees=[45, 45])]
    Trsf_2 = [transforms.RandomAffine(degrees=[90, 90])]

    list_transf = [Trsf_0, Trsf_1, Trsf_2]

    scenario = TransformationIncremental(
        cl_dataset=dummy, incremental_transformations=list_transf
    )

    ref_data = None
    raw_ref_data = None
    for task_id, taskset in enumerate(scenario):

        samples, _, _ = taskset.get_random_samples(10)
        # we need raw data to apply same transformation as the TransformationIncremental class
        raw_samples, _, _ = taskset.get_raw_samples(range(10))

        if task_id == 0:
            ref_data = samples
            raw_ref_data = raw_samples
        else:
            # we verify that data has changed
            assert not torch.all(ref_data.eq(samples))

            assert (raw_samples == raw_ref_data
                    ).all()  # raw data should be the same in this scenario

            # we test transformation on one data point and verify if it is applied
            trsf = list_transf[task_id][0]
            raw_sample = Image.fromarray(raw_ref_data[0].astype("uint8"))
            trsf_data = trsf(raw_sample)
            trsf_data = transforms.ToTensor()(trsf_data)

            assert torch.all(trsf_data.eq(samples[0]))


'''
Test the initialization with three tasks with degree range
'''


def test_init_range(numpy_data):
    x, y = numpy_data
    dummy = InMemoryDataset(x, y)

    Trsf_0 = []
    Trsf_1 = [transforms.RandomAffine(degrees=[40, 50])]
    Trsf_2 = [transforms.RandomAffine(degrees=[85, 95])]

    list_transf = [Trsf_0, Trsf_1, Trsf_2]

    scenario = TransformationIncremental(
        cl_dataset=dummy, incremental_transformations=list_transf
    )


@pytest.mark.parametrize("shared_label_space", [False, True])
def test_init_shared_label_space(numpy_data, shared_label_space):
    x, y = numpy_data
    dummy = InMemoryDataset(x, y)

    Trsf_0 = []
    Trsf_1 = [transforms.RandomAffine(degrees=[40, 50])]
    Trsf_2 = [transforms.RandomAffine(degrees=[85, 95])]

    dummy_transf = [Trsf_0, Trsf_1, Trsf_2]

    scenario = TransformationIncremental(
        cl_dataset=dummy,
        incremental_transformations=dummy_transf,
        shared_label_space=shared_label_space
    )

    for task_id, taskset in enumerate(scenario):
        assert taskset.nb_classes == NB_CLASSES
        classes = taskset.get_classes()
        if shared_label_space:
            assert classes.max() == NB_CLASSES - 1
            assert classes.min() == 0
        else:
            assert classes.max() == (NB_CLASSES * (task_id + 1)) - 1
            assert classes.min() == (NB_CLASSES * task_id)


def test_get_task_transformation(numpy_data):
    x, y = numpy_data
    dummy = InMemoryDataset(x, y)

    Trsf_0 = []
    Trsf_1 = [transforms.RandomAffine(degrees=[40, 50])]
    Trsf_2 = [transforms.RandomAffine(degrees=[85, 95])]

    dummy_transf = [Trsf_0, Trsf_1, Trsf_2]

    base_transformations = [
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ]

    scenario = TransformationIncremental(
        cl_dataset=dummy,
        incremental_transformations=dummy_transf,
        base_transformations=base_transformations
    )

    for task_id, taskset in enumerate(scenario):
        # first task specific transformation then base_transformation
        tot_transf_task = transforms.Compose(dummy_transf[task_id] + base_transformations)

        # we compare the str representation of the composition
        assert tot_transf_task.__repr__() == scenario.get_task_transformation(task_id).__repr__()


def test_init_fail2(numpy_data):
    train = numpy_data
    dummy = InMemoryDataset(*train)

    # No transformation is set
    with pytest.raises(TypeError):
        scenario = TransformationIncremental(cl_dataset=dummy)
