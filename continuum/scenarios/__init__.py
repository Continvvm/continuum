# pylint: disable=C0401
# flake8: noqa
from continuum.scenarios.base import _BaseScenario
from continuum.scenarios.continual_scenario import ContinualScenario
from continuum.scenarios.class_incremental import ClassIncremental
from continuum.scenarios.instance_incremental import InstanceIncremental
from continuum.scenarios.transformation_incremental import TransformationIncremental
from continuum.scenarios.rotations import Rotations
from continuum.scenarios.permutations import Permutations
from continuum.scenarios.segmentation import SegmentationClassIncremental

__all__ = [
    "ContinualScenario",
    "ClassIncremental",
    "InstanceIncremental",
    "Rotations",
    "Permutations",
    "TransformationIncremental",
    "SegmentationClassIncremental"
]

