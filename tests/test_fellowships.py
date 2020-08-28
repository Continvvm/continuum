import numpy as np
import pytest
import os

from continuum.datasets import MNIST
from continuum.datasets import CIFARFellowship, MNISTFellowship

@pytest.mark.slow
def test_MNIST_Fellowship():
    cl_dataset = MNISTFellowship(data_path="./pytest/Samples/Datasets", train=True, download=True)


@pytest.mark.slow
def test_CIFAR_Fellowship():
    cl_dataset = CIFARFellowship(data_path="./pytest/Samples/Datasets", train=True, download=True)