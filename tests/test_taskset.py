import numpy as np
import pytest

from continuum.tasks import TaskSet, split_train_val


@pytest.mark.parametrize("val_split,nb_val", [(0., 0), (0.1, 1), (0.8, 8), (0.99, 9), (1.0, 10)])
def test_split_train_val(val_split, nb_val):
    """
    Return train set.

    Args:
        val_split: (todo): write your description
        nb_val: (todo): write your description
    """
    x = np.random.rand(10, 2, 2, 3)
    y = np.ones((10,))
    t = np.ones((10,))

    base_set = TaskSet(x, y, t, None)

    train_set, val_set = split_train_val(base_set, val_split)
    assert len(val_set) == nb_val
    assert len(train_set) + len(val_set) == len(base_set)


@pytest.mark.parametrize("nb_samples", [1, 5, 10])
def test_get_random_samples(nb_samples):
    """
    Return a set of samples.

    Args:
        nb_samples: (int): write your description
    """
    x = np.ones((10, 2, 2, 3))
    y = np.ones((10,))
    t = np.ones((10,))

    base_set = TaskSet(x, y, t, None)

    base_set.get_random_samples(nb_samples=nb_samples)


@pytest.mark.parametrize("nb_samples", [1, 5, 10])
def test_get_raw_samples(nb_samples):
    """
    Get raw samples.

    Args:
        nb_samples: (int): write your description
    """
    x = np.ones((10, 2, 2, 3))
    y = np.ones((10,))
    t = np.ones((10,))

    base_set = TaskSet(x, y, t, None)

    data, y_, t_ = base_set.get_raw_samples(indexes=range(nb_samples))

    assert (x[:nb_samples] == data).all()
    assert (y[:nb_samples] == y_).all()
    assert (t[:nb_samples] == t_).all()
