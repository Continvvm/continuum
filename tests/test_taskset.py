import numpy as np
import pytest

from continuum import TaskSet, split_train_val


@pytest.mark.parametrize("val_split,nb_val",
[
    (0., 0),
    (0.1, 1),
    (0.8, 8),
    (0.99, 9),
    (1.0, 10)
])
def test_split_train_val(val_split, nb_val):
    x = np.ones((10, 2, 2, 3))
    y = np.ones((10,))
    t = np.ones((10,))

    base_set = TaskSet(x, y, t, None)

    train_set, val_set = split_train_val(base_set, val_split)
    assert len(val_set) == nb_val
    assert len(train_set) + len(val_set) == len(base_set)
