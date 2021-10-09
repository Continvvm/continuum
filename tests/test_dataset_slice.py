import numpy as np
import pytest
from continuum.datasets import InMemoryDataset


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

    return InMemoryDataset(x, y, t)


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




