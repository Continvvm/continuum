import numpy as np
import pytest
from torch.utils.data import DataLoader


from continuum.scenarios import ClassIncremental
from continuum.datasets import MNIST, CIFAR10, CIFAR100, KMNIST, FashionMNIST, InMemoryDataset
from torchvision.transforms import transforms


# yapf: disable


@pytest.mark.slow
@pytest.mark.parametrize("dataset, increment", [(MNIST, 5),
                                                (KMNIST, 2),
                                                (FashionMNIST, 1),
                                                (CIFAR10, 2),
                                                (CIFAR100, 10)])
def test_with_dataset_simple_increment(tmpdir, dataset, increment):
    dataset = dataset(data_path=tmpdir, download=True, train=True)
    scenario = ClassIncremental(cl_dataset=dataset,
                                increment=increment,
                                transformations=[transforms.ToTensor()]
                                )

    for task_id, taskset in enumerate(scenario):
        classes = taskset.get_classes()

        assert len(classes) == increment

        # check if there is continuity in classes by default
        assert len(classes) == (classes.max() - classes.min() + 1)


@pytest.mark.slow
@pytest.mark.parametrize("dataset, increment", [(MNIST, [5, 1, 1, 3]),
                                                (KMNIST, [2, 2, 4, 2]),
                                                (FashionMNIST, [1, 2, 1, 2, 1, 2, 1]),
                                                (CIFAR10, [2, 2, 2, 2, 2]),
                                                (CIFAR100, [50, 10, 20, 20])])
def test_with_dataset_composed_increment(tmpdir, dataset, increment):
    dataset = dataset(data_path=tmpdir, download=True, train=True)
    scenario = ClassIncremental(cl_dataset=dataset,
                                increment=increment,
                                transformations=[transforms.ToTensor()]
                                )

    for task_id, taskset in enumerate(scenario):
        classes = taskset.get_classes()

        assert len(classes) == increment[task_id]

        # check if there is continuity in classes by default
        assert len(classes) == (classes.max() - classes.min() + 1)


@pytest.fixture
def fake_data():
    x_train = np.random.randint(0, 255, size=(20, 32, 32, 3))
    y_train = []
    for i in range(10):
        y_train.append(np.ones(2) * i)
    y_train = np.concatenate(y_train)

    return InMemoryDataset(x_train, y_train)


@pytest.mark.parametrize("class_order", [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    [3, 9, 2, 0, 4, 5, 9, 7, 6, 1],
])
def test_taskid(fake_data, class_order):
    scenario = ClassIncremental(
        cl_dataset=fake_data,
        increment=2
    )

    for task_id, taskset in enumerate(scenario):
        loader = DataLoader(taskset, batch_size=32)

        for x, y, t in loader:
            assert t[0].item() == task_id
            assert (t == task_id).all()
