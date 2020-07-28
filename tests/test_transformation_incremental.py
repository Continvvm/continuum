import numpy as np
import pytest
from continuum.scenarios import TransformationIncremental
from continuum.datasets import InMemoryDataset
from torchvision.transforms import transforms
from PIL import Image
import torch

@pytest.fixture
def numpy_data():
    nb_classes = 6
    nb_data = 10

    x_train = []
    y_train = []
    x_train.append(np.array([np.random.randint(100, size=(2, 2, 3)).astype(dtype=np.uint8)] * nb_data))
    y_train.append(np.random.randint(nb_classes, size=(nb_data)))
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
    Trsf_1 = [transforms.RandomAffine(degrees=[45,45])]
    Trsf_2 = [transforms.RandomAffine(degrees=[90,90])]

    list_transf = [Trsf_0, Trsf_1, Trsf_2]

    continuum = TransformationIncremental(cl_dataset=dummy, nb_tasks=3, incremental_transformations=list_transf)

    ref_data = None
    raw_ref_data = None
    for task_id, train_dataset in enumerate(continuum):

        samples, _, _ = train_dataset.rand_samples(10)
        # we need raw data to apply same transformation as the TransformationIncremental class
        raw_samples, _, _ = train_dataset.get_raw_samples_from_ind(range(10))

        if task_id == 0:
            ref_data = samples
            raw_ref_data = raw_samples
        else:
            # we verify that data has changed
            assert not torch.all(ref_data.eq(samples))

            assert (raw_samples==raw_ref_data).all() # raw data should be the same in this scenario

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

    continuum = TransformationIncremental(cl_dataset=dummy, nb_tasks=3, incremental_transformations=list_transf)

def test_init_fail(numpy_data):
    train = numpy_data
    dummy = InMemoryDataset(*train)

    Trsf_0 = []
    Trsf_1 = [transforms.RandomAffine(degrees=[40, 50])]
    Trsf_2 = [transforms.RandomAffine(degrees=[85, 95])]

    list_transf = [Trsf_0, Trsf_1, Trsf_2]

    with pytest.raises(ValueError):
        TransformationIncremental(cl_dataset=dummy, nb_tasks=2, incremental_transformations=list_transf)

def test_init_fail2(numpy_data):
    train = numpy_data
    dummy = InMemoryDataset(*train)

    # No transformation is set
    with pytest.raises(TypeError):
        clloader = TransformationIncremental(cl_dataset=dummy, nb_tasks=3)
