# pylint: disable=C0401
# flake8: noqa
from continuum.scenarios.base import _BaseCLLoader
from continuum.scenarios.class_incremental import ClassIncremental
from continuum.scenarios.data_incremental import DataIncremental
from continuum.scenarios.instance_incremental import InstanceIncremental

__all__ = ["ClassIncremental", "DataIncremental", "InstanceIncremental"]
