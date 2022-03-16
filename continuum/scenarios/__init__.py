# pylint: disable=C0401
# flake8: noqa
from continuum.scenarios.base import _BaseScenario
from continuum.scenarios.continual_scenario import ContinualScenario
from continuum.scenarios.class_incremental import ClassIncremental
from continuum.scenarios.instance_incremental import InstanceIncremental
from continuum.scenarios.specific_scenarios import ALMA
from continuum.scenarios.transformation_incremental import TransformationIncremental
from continuum.scenarios.rotations import Rotations
from continuum.scenarios.permutations import Permutations
from continuum.scenarios.segmentation import SegmentationClassIncremental
from continuum.scenarios.hashed import HashedScenario
from continuum.scenarios.online_fellowship import OnlineFellowship
from continuum.scenarios import hf

from continuum.scenarios.scenario_utils import create_subscenario, encode_scenario, remap_class_vector, get_scenario_remapping

__all__ = [
    "ContinualScenario",
    "ClassIncremental",
    "InstanceIncremental",
    "Rotations",
    "Permutations",
    "TransformationIncremental",
    "SegmentationClassIncremental",
    "HashedScenario",
    "OnlineFellowship",
    "hf",
    "ALMA"
]
