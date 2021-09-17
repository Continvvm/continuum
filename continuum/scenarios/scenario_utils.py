import numpy as np
import torch
from continuum.datasets import InMemoryDataset
from continuum.scenarios import ContinualScenario


def create_subscenario(base_scenario, task_indexes):
    """
    In this function we want to create a subscenario from the different tasks, either by subsampling tasks or reodering
    or both.
    """

    if torch.is_tensor(task_indexes):
        task_indexes = task_indexes.numpy()

    new_x, new_y, new_t = None, None, None
    if base_scenario.cl_dataset.bounding_boxes is not None:
        raise ValueError("the function create_subscenario is not compatible with scenario with bounding_boxes yet.")

    for i, index in enumerate(task_indexes):
        taskset = base_scenario[index]
        all_task_indexes = np.arange(len(taskset))
        x, y, t = taskset.get_raw_samples(all_task_indexes)
        t = np.ones(len(y)) * i
        if new_x is None:
            new_x = x
            new_y = y
            new_t = t
        else:
            new_x = np.concatenate([new_x, x], axis=0)
            new_y = np.concatenate([new_y, y], axis=0)
            new_t = np.concatenate([new_t, t], axis=0)
    dataset = InMemoryDataset(new_x, new_y, new_t, data_type=base_scenario.cl_dataset.data_type)

    return ContinualScenario(dataset, transformations=base_scenario.transformations)
