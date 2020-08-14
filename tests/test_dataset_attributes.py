import pytest

from continuum import datasets as cont_datasets

ATTRS = ["get_data", "_download"]


@pytest.mark.parametrize("dataset_name", [d for d in dir(cont_datasets) if d[0].isupper()])
def test_has_attr(dataset_name):
    d = getattr(cont_datasets, dataset_name)

    for attr in ATTRS:
        assert hasattr(d, attr), (dataset_name, attr)
