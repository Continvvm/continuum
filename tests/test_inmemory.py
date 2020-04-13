import numpy as np
import pytest

from clloader import CLLoader
from clloader.datasets import InMemoryDataset
from torch.utils.data import DataLoader


def gen_data():
    x_train = np.random.randint(0, 255, size=(20, 32, 32, 3))
    y_train = []
    for i in range(10):
        y_train.append(np.ones(2) * i)
    y_train = np.concatenate(y_train)

    x_test = np.random.randint(0, 255, size=(20, 32, 32, 3))
    y_test = np.copy(y_train)

    return (x_train, y_train), (x_test, y_test)


# yapf: disable

@pytest.mark.parametrize("increment,initial_increment,nb_tasks", [
    (2, 0, 5),
    (5, 0, 2),
    (1, 5, 6),
    (2, 4, 4),
    ([5, 1, 1, 3], 0, 4)
])
def test_increments(increment, initial_increment, nb_tasks):
    train, test = gen_data()
    dummy = InMemoryDataset(*train, *test)
    clloader = CLLoader(dummy, increment, initial_increment)

    assert clloader.nb_tasks == nb_tasks
    seen_tasks = 0
    for train_dataset, test_dataset in clloader:
        seen_tasks += 1

        for _ in DataLoader(train_dataset):
            pass
        for _ in DataLoader(test_dataset):
            pass

    assert seen_tasks == nb_tasks
