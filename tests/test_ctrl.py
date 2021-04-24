import os

import pytest
import numpy as np

from continuum import ContinualScenario
from continuum.datasets import CTRLplus, CTRLminus, CTRLout, CTRLin, CTRLplastic


DATA_PATH = os.environ.get("CONTINUUM_DATA_PATH")


@pytest.mark.slow
@pytest.mark.parametrize("dataset,nb_tasks,classes_per_task,qt_train,qt_val", [
    (CTRLminus, 6, [
        list(range(10)), list(range(10, 20)), list(range(20, 67)),
        list(range(67, 77)), list(range(77, 87)), list(range(10))],
        [4000, 400, 376, 400, 400, 400],
        [2000, 200, 188, 200, 200, 200]
    ),
    (CTRLplus, 6, [
        list(range(10)), list(range(10, 20)), list(range(20, 67)),
        list(range(67, 77)), list(range(77, 87)), list(range(10))],
        [400, 400, 376, 400, 400, 4000],
        [200, 200, 188, 200, 200, 2000]
    ),
    (CTRLin, 6, [
        list(range(10)), list(range(10, 20)), list(range(20, 67)),
        list(range(67, 77)), list(range(77, 87)), list(range(10))],
        [4000, 400, 376, 400, 400, 50],
        [2000, 200, 188, 200, 200, 30]
    ),
    (CTRLout, 6, [
        list(range(10)), list(range(10, 20)), list(range(20, 67)),
        list(range(67, 77)), list(range(77, 87)), list(range(87, 97))],
        [4000, 400, 376, 400, 400, 400],
        [2000, 200, 188, 200, 200, 200]
    ),
    (CTRLplastic, 5, [
        list(range(10)), list(range(10, 57)),
        list(range(57, 67)), list(range(67, 77)), list(range(77, 87))],
        [400, 376, 400, 400, 4000],
        [200, 188, 200, 200, 2000]
    ),
])
def test_ctrl(tmpdir, dataset, nb_tasks, classes_per_task, qt_train, qt_val):
    path = DATA_PATH or tmpdir  # Use env variable else pytest default temp dir

    s_train = ContinualScenario(dataset(path, split="train", download=True))
    s_val = ContinualScenario(dataset(path, split="val", download=True))
    s_test = ContinualScenario(dataset(path, split="test", download=True))

    assert len(s_train) == len(s_val) == len(s_test) == nb_tasks

    for i, (tr_set, va_set, te_set) in enumerate(zip(s_train, s_val, s_test)):
        assert np.unique(tr_set._y).tolist() == np.unique(va_set._y).tolist() == np.unique(te_set._y).tolist()
        assert np.unique(tr_set._y).tolist() == classes_per_task[i], i
        assert len(tr_set) == qt_train[i]
        assert len(va_set) == qt_val[i]
