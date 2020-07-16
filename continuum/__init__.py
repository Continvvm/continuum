"""Continuum lib.

Made by Arthur Douillard and Timothee Lesort.

The goal of this library is to provide clean and simple to use utilities for
Continual Learning.
"""
# pylint: disable=C0401
# flake8: noqa
from continuum import datasets
from continuum.scenarios import *
from continuum.task_set import TaskSet, split_train_val
from continuum.viz import plot_samples
