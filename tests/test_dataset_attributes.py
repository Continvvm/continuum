import pytest

from continuum import datasets as cont_datasets
from torchvision.datasets import EMNIST, KMNIST
from continuum.datasets import PyTorchDataset

ATTRS = ["get_data", "_download"]


@pytest.mark.parametrize("dataset_name", [d for d in dir(cont_datasets) if d[0].isupper()])
def test_has_attr(dataset_name):
    d = getattr(cont_datasets, dataset_name)

    for attr in ATTRS:
        assert hasattr(d, attr), (dataset_name, attr)


@pytest.mark.slow
def test_PytorchDataset_EMNIST(tmpdir):
    dataset_train = PyTorchDataset(tmpdir, dataset_type=EMNIST, train=True, download=True, split='letters')


@pytest.mark.slow
def test_PytorchDataset_KMNIST(tmpdir):
    dataset_train = PyTorchDataset(tmpdir, dataset_type=KMNIST, train=True, download=True)