import pytest
import numpy as np
from torchvision.transforms import Resize, ToTensor

from continuum.datasets import Stream51
from continuum.scenarios import InstanceIncremental
from continuum.scenarios import ClassIncremental



'''
Test the visualization with instance_class scenario
'''
@pytest.mark.slow
def test_scenario_clip_ClassIncremental(tmpdir):

    dataset = Stream51('../Datasets', task_criterion="clip")
    # dataset = Stream51('../Datasets', task_criterion="video")
    # scenario = InstanceIncremental(dataset, transformations=[Resize((224, 224)), ToTensor()])
    scenario = ClassIncremental(dataset, increment=1, transformations=[Resize((224, 224)), ToTensor()])

    print(f"Nb classes : {scenario.nb_classes} ")
    print(f"Nb tasks : {scenario.nb_tasks} ")
    for task_id, task_set in enumerate(scenario):
        print(f"Task {task_id} : {task_set.nb_classes} classes")
        task_set.plot(path="Archives/Samples/Stream51/CI",
                      title="Stream51_InstanceIncremental_{}.jpg".format(task_id),
                      nb_samples=100)



'''
Test the visualization with instance scenario
'''
@pytest.mark.slow
def test_scenario_clip_InstanceIncremental(tmpdir):

    dataset = Stream51('../Datasets', task_criterion="clip")
    scenario = InstanceIncremental(dataset, transformations=[Resize((224, 224)), ToTensor()])

    print(f"Nb classes : {scenario.nb_classes} ")
    print(f"Nb tasks : {scenario.nb_tasks} ")
    for task_id, task_set in enumerate(scenario):
        print(f"Task {task_id} : {task_set.nb_classes} classes")
        task_set.plot(path="Archives/Samples/Stream51/CI",
                      title="Stream51_InstanceIncremental_{}.jpg".format(task_id),
                      nb_samples=100)
