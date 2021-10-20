import os

import numpy as np
import pytest
import h5py

from continuum.datasets import InMemoryDataset, H5Dataset


@pytest.fixture
def dataset():
    x = np.zeros((20, 4, 4, 3))
    y = np.zeros((20,))
    t = np.zeros((20,))

    for i in range(20):
        x[i] = i

    c = 0
    for i in range(0, 20, 2):
        y[i] = c
        y[i+1] = c
        c += 1
    for i in range(0, 20, 2):
        t[i] = 1

    return x, y, t


@pytest.mark.parametrize("keep_classes,discard_classes,keep_tasks,discard_tasks,error,ids", [
    ([1], [1], None, None, True, None),
    (None, None, [1], [1], True, None),
    (list(range(10)), None, None, None, False, list(range(20))),
    ([0, 1], None, None, None, False, [0, 1, 2, 3]),
    (None, [0, 1], None, None, False, list(range(4, 20))),
    (None, None, [0, 1], None, False, list(range(20))),
    (None, None, [1], None, False, list(range(0, 20, 2))),
    (None, None, None, [1], False, list(range(1, 20, 2))),
    ([0, 1], None, [1], None, False, [0, 2]),
])
def test_slice(
        dataset,
        keep_classes, discard_classes,
        keep_tasks, discard_tasks,
        error,
        ids
    ):

    dataset = InMemoryDataset(*dataset)

    if error:
        with pytest.raises(Exception):
            sliced_dataset = dataset.slice(
                keep_classes, discard_classes,
                keep_tasks, discard_tasks
            )
        return
    else:
        sliced_dataset = dataset.slice(
            keep_classes, discard_classes,
            keep_tasks, discard_tasks
        )

    x, _, _ = sliced_dataset.get_data()

    assert (np.unique(x) == np.array(ids)).all(), (np.unique(x), ids)


@pytest.mark.parametrize("keep_classes,discard_classes,keep_tasks,discard_tasks,error,ids", [
    ([1], [1], None, None, True, None),
    (None, None, [1], [1], True, None),
    (list(range(10)), None, None, None, False, list(range(20))),
    ([0, 1], None, None, None, False, [0, 1, 2, 3]),
    (None, [0, 1], None, None, False, list(range(4, 20))),
    (None, None, [0, 1], None, False, list(range(20))),
    (None, None, [1], None, False, list(range(0, 20, 2))),
    (None, None, None, [1], False, list(range(1, 20, 2))),
    ([0, 1], None, [1], None, False, [0, 2]),
])
def test_slice_h5(
        tmpdir,
        dataset,
        keep_classes, discard_classes,
        keep_tasks, discard_tasks,
        error,
        ids
    ):

    dataset = H5Dataset(*dataset, data_path=os.path.join(tmpdir, "test.h5"))

    if error:
        with pytest.raises(Exception):
            sliced_dataset = dataset.slice(
                os.path.join(tmpdir, "test_bis.h5"),
                keep_classes, discard_classes,
                keep_tasks, discard_tasks
            )
        return
    else:
        sliced_dataset = dataset.slice(
            os.path.join(tmpdir, "test_bis.h5"),
            keep_classes, discard_classes,
            keep_tasks, discard_tasks
        )

    h5_path, _, _ = sliced_dataset.get_data()

    assert h5_path == os.path.join(tmpdir, "test_bis.h5")
    with h5py.File(h5_path, 'r') as hf:
        x = hf['x'][:]

    assert (np.unique(x) == np.array(ids)).all(), (np.unique(x), ids)


