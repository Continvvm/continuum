import numpy as np
import pytest
import os
from torchvision import transforms

from continuum.datasets import TinyImageNet200
from continuum.scenarios import ClassIncremental

DATA_PATH = os.environ.get("CONTINUUM_DATA_PATH")

@pytest.mark.slow
@pytest.mark.parametrize("dataset, name, shape", [(TinyImageNet200, "TinyImageNet200", [64, 64, 3])])
def test_slow_datasets(dataset, name, shape):
    cl_dataset_train = dataset(data_path=DATA_PATH, download=False, train=True)
    cl_dataset_test = dataset(data_path=DATA_PATH, download=False, train=False)

    assert cl_dataset_train.num_classes == cl_dataset_test.num_classes == 200