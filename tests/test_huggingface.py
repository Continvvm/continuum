


import pytest
import datasets

from continuum.scenarios.hf import HuggingFaceFellowship, HuggingFaceContinual


class _Dataset:
    def __init__(self, rows=None):
        if rows:
            self.rows = rows
        else:
            self.rows = [
                {'genre': 'sf', 'b': i}
                for i in range(10)
            ] + [
                {'genre': 'fantasy', 'b': i}
                for i in range(10)
            ] + [
                {'genre': 'bio', 'b': i}
                for i in range(10)
            ] + [
                {'genre': 'history', 'b': i}
                for i in range(10)
            ]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, column_name):
        return [row[column_name] for row in self.rows]

    def filter(self, lbd):
        return _Dataset(list(filter(lbd, self.rows)))


def mock_incremental_dataset(*args, **kwargs):
    return _Dataset()


@pytest.mark.parametrize("increment", [1, 2, 4])
def test_hf_incremental(mocker, increment):
    mocker.patch.object(datasets, "load_dataset", new=mock_incremental_dataset)

    scenario = HuggingFaceContinual(
        "foo", split_field="genre", increment=increment
    )

    assert scenario.nb_classes == 4
    assert scenario.nb_samples == 40
    assert len(scenario) == 4 // increment

    for taskset in scenario:
        classes = set([row["genre"] for row in taskset.rows])
        assert len(classes) == increment


def test_hf_fellowship():
    scenario = HuggingFaceFellowship(
        [_Dataset(), _Dataset()]
    )

    with pytest.raises(NotImplementedError):
        scenario.nb_classes
    assert scenario.nb_samples == 80
    assert len(scenario) == 2

    for taskset in scenario:
        assert len(taskset) == 40
