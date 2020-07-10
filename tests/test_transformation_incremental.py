import numpy as np
import pytest
from continuum.scenarios import TransformationIncremental
from continuum.datasets import InMemoryDataset
from torchvision.transforms import transforms

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

    return x_train, y_train.astype(int)

'''
Test the initialization with three tasks
'''
def test_init(numpy_data):
    x, y = numpy_data
    dummy = InMemoryDataset(x, y, train='train')

    Trsf_0 = []
    Trsf_1 = [transforms.RandomAffine(degrees=[40, 50])]
    Trsf_2 = [transforms.RandomAffine(degrees=[85, 95])]

    list_transf = [Trsf_0, Trsf_1, Trsf_2]

    continuum = TransformationIncremental(cl_dataset=dummy, nb_tasks=3, incremental_transformations=list_transf)

@pytest.mark.xfail
def test_init_fail(numpy_data):
    train, test = numpy_data
    dummy = TransformationIncremental(*train, *test)

    Trsf_0 = []
    Trsf_1 = [transforms.RandomAffine(degrees=[40, 50])]
    Trsf_2 = [transforms.RandomAffine(degrees=[85, 95])]

    list_transf = [Trsf_0, Trsf_1, Trsf_2]

    # the wrong number of task is set
    clloader = TransformationIncremental(cl_dataset=dummy, nb_tasks=2, incremental_transformations=list_transf)
