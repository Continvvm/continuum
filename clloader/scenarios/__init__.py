# pylint: disable=C0401
# flake8: noqa
from clloader.scenarios.base import _BaseCLLoader
from clloader.scenarios.class_incremental import ClassIncremental
from clloader.scenarios.data_incremental import DataIncremental
from clloader.scenarios.instance_incremental import InstanceIncremental

__all__ = ["ClassIncremental", "DataIncremental", "InstanceIncremental"]
