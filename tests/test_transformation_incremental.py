import numpy as np
import pytest
from continuum.scenarios import TransformationIncremental
from continuum.datasets import InMemoryDataset
from torchvision.transforms import transforms

@pytest.fixture
def numpy_data():

    nb_classes = 6
    nb_data = 10

    x_train = []
    y_train = []
    for i in range(nb_classes):
        x_train.append(np.expand_dims(np.array([np.eye(5, dtype=np.uint8)]*nb_data), axis=-1) * i)
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
    Trsf_1 = [transforms.RandomAffine(degrees=[45,45])]
    Trsf_2 = [transforms.RandomAffine(degrees=[90,90])]

    list_transf = [Trsf_0, Trsf_1, Trsf_2]

    continuum = TransformationIncremental(cl_dataset=dummy, nb_tasks=3, incremental_transformations=list_transf)

    ref_data = None
    for task_id, train_dataset in enumerate(continuum):

        samples = train_dataset.rand_samples(1)

        print("yooooooooooooooooooooooo")
        print(samples.shape)

        if task_id == 0:
            ref_data = samples
        else:
            assert not (ref_data==samples).all()

            trsf_data = list_transf[task_id](ref_data)
            assert not (ref_data==samples).all()



'''
Test the initialization with three tasks with degree range
'''
def test_init_range(numpy_data):
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
    dummy = TransformationIncremental(*train)

    Trsf_0 = []
    Trsf_1 = [transforms.RandomAffine(degrees=[40, 50])]
    Trsf_2 = [transforms.RandomAffine(degrees=[85, 95])]

    list_transf = [Trsf_0, Trsf_1, Trsf_2]

    # the wrong number of task is set
    clloader = TransformationIncremental(cl_dataset=dummy, nb_tasks=2, incremental_transformations=list_transf)

@pytest.mark.xfail
def test_init_fail2(numpy_data):
    train, test = numpy_data
    dummy = TransformationIncremental(*train)

    # No transformation is set
    clloader = TransformationIncremental(cl_dataset=dummy, nb_tasks=3)
