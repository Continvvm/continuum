import os
import numpy as np
import torch
from torch.utils.data import DataLoader

from continuum.datasets import InMemoryDataset, H5Dataset
from continuum.scenarios import ContinualScenario, OnlineFellowship
from continuum.tasks import TaskType


def create_subscenario(base_scenario, task_indexes):
    """
    In this function we want to create a subscenario from the different tasks, either by subsampling tasks or reodering
    or both.
    """

    if torch.is_tensor(task_indexes):
        task_indexes = task_indexes.numpy()

    if base_scenario.transformations is not None and isinstance(base_scenario.transformations[0], list):
        transformations = [base_scenario.transformations[i] for i in task_indexes]
    else:
        transformations = base_scenario.transformations
    sub_scenario = None
    if isinstance(base_scenario, OnlineFellowship):
        # We just want to changes base_scenario.cl_datasets order
        new_cl_datasets = [base_scenario.cl_datasets[i] for i in task_indexes]
        sub_scenario = OnlineFellowship(new_cl_datasets,
                                        transformations=transformations,
                                        update_labels=base_scenario.update_labels)
    elif base_scenario.cl_dataset.data_type == TaskType.H5:
        list_taskset = [base_scenario[i] for i in task_indexes]
        sub_scenario = OnlineFellowship(list_taskset,
                                        transformations=transformations,
                                        update_labels=False)
    else:
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
        sub_scenario = ContinualScenario(dataset, transformations=transformations)

    return sub_scenario


@torch.no_grad()
def encode_into_dataset(model, scenario, batch_size, filename, inference_fct=None):
    """This function encode a scenario into a h5 dataset to reproduce the same scenario with features.

    :param model: model to encode the data.
    :param scenario: scenario to encode.
    :param batch_size: batch size to load data.
    :param filename: filename for the h5 dataset.
    :param inference_fct: A function that make possible to have a sophisticate way to get features.
    """
    training_mode = model.training

    if inference_fct is None:
        inference_fct = (lambda model, x: model.to(torch.device('cuda:0'))(x.to(torch.device('cuda:0'))))

    # we save feature in eval mode
    model.eval()

    encoded_dataset = None
    for task_id, taskset in enumerate(scenario):
        # we need to load the data to use the transformation if there is some
        loader = DataLoader(taskset, shuffle=False, batch_size=batch_size)
        for i, (x, y, t) in enumerate(loader):
            features = inference_fct(model, x)
            if t is None:
                t = (torch.ones(len(y)) * task_id).long()

            if task_id == 0 and i == 0:
                encoded_dataset = H5Dataset(features.cpu().numpy(), y, t, data_path=filename)
            else:
                encoded_dataset.add_data(features.cpu().numpy(), y, t)

    model.train(training_mode)
    return encoded_dataset


def encode_scenario(scenario, model, batch_size, filename, inference_fct=None):
    """This function created an encoded scenario dataset and convert it into a ContinualScenario.

    :param model: model to encode the data.
    :param scenario: scenario to encode.
    :param batch_size: batch size to load data.
    :param filename: filename for the h5 dataset.
    :param inference_fct: A function that make possible to have a sophisticate way to get features.
    """

    if os.path.isfile(filename):
        raise ValueError(f"File name: {filename} already exists")

    print(f"Encoding {filename}.")
    encoded_dataset = encode_into_dataset(model, scenario, batch_size, filename, inference_fct)
    print(f"Encoding is done.")

    return ContinualScenario(encoded_dataset)
