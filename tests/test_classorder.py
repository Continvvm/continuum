import numpy as np
import pytest
from torch.utils.data import DataLoader

from continuum.datasets import InMemoryDataset
from continuum.scenarios import ClassIncremental


class InMemoryDatasetTest(InMemoryDataset):

    def __init__(self, *args, class_order=None, **kwargs):
        """
        Initialize the class.

        Args:
            self: (todo): write your description
            class_order: (todo): write your description
        """
        super().__init__(*args, **kwargs)
        self._class_order = class_order

    @property
    def class_order(self):
        """
        Return the order.

        Args:
            self: (todo): write your description
        """
        return self._class_order


@pytest.fixture
def numpy_data():
    """
    Concatenate the training data.

    Args:
    """
    x_train = []
    y_train = []
    for i in range(10):
        x_train.append(np.ones((5, 4, 4, 3), dtype=np.uint8) * i)
        y_train.append(np.ones(5) * i)
    x_train = np.concatenate(x_train)
    y_train = np.concatenate(y_train)

    x_test = np.copy(x_train)
    y_test = np.copy(y_train)

    return (x_train, y_train.astype(int)), (x_test, y_test.astype(int))


# yapf: disable

@pytest.mark.parametrize("classes,default_class_order,class_order", [
    ([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]], None, None),
    ([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]], list(range(10)), None),
    ([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]], None, list(range(10))),
    ([[1, 3, 5, 7, 9], [0, 2, 4, 6, 8]], [1, 3, 5, 7, 9, 0, 2, 4, 6, 8], None),
    ([[1, 3, 5, 7, 9], [0, 2, 4, 6, 8]], None, [1, 3, 5, 7, 9, 0, 2, 4, 6, 8]),
    ([[1, 3, 5, 7, 9], [0, 2, 4, 6, 8]], list(range(10)), [1, 3, 5, 7, 9, 0, 2, 4, 6, 8])
])
def test_increments(numpy_data, classes, default_class_order, class_order):
    """
    Test for the test tasks.

    Args:
        numpy_data: (int): write your description
        classes: (todo): write your description
        default_class_order: (str): write your description
        class_order: (str): write your description
    """
    train, test = numpy_data
    dummy = InMemoryDatasetTest(*train, class_order=default_class_order)
    scenario = ClassIncremental(dummy, 2, 5, class_order=class_order)

    gt_new_targets = [np.arange(5), np.arange(5) + 5]
    for task_id, taskset in enumerate(scenario):
        for _ in DataLoader(taskset):
            pass

        unique_classes = np.sort(np.unique(taskset._x))
        ans = (unique_classes == np.array(classes[task_id]))
        assert ans.all(), (task_id, unique_classes, np.array(classes[task_id]))

        original_targets = np.sort(np.unique(scenario.get_original_targets(taskset._y)))
        ans = (original_targets == np.array(classes[task_id]))
        assert ans.all(), (task_id, original_targets, np.array(classes[task_id]))

        new_targets = np.sort(np.unique(taskset._y))
        ans = (new_targets == gt_new_targets[task_id])
        assert ans.all(), (task_id, new_targets, gt_new_targets[task_id])
