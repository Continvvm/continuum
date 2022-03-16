import warnings
from typing import Callable, List, Optional, Union

from continuum.datasets import _ContinuumDataset
from continuum.scenarios import InstanceIncremental

import numpy as np

class ALMA(InstanceIncremental):
    """ALMA Scenario, generating tasks by randomly partioning a fixed dataset.
    NOTE: this is similar to InstanceIncremental, but reformatted with the
    language of the paper https://arxiv.org/abs/2106.09563. The main difference is
    that per class samples are not equally split across tasks, but rather selected
    randomly from the full subset.

    Scenario: Classes are always the same but instances change (NI scenario)

    :param cl_dataset: A continual dataset.
    :param nb_megabatches: The scenario number of megabatches (chunks, partitions)
    :param transformations: A list of transformations applied to all tasks. If
                            it's a list of list, then the transformation will be
                            different per task.
    :param random_seed: A random seed to init random processes.
    """
    def __init__(
            self,
            cl_dataset: _ContinuumDataset,
            nb_megabatches: Optional[int] = None,
            transformations: Union[List[Callable], List[List[Callable]]] = None,
            random_seed: int = 1
    ):

        if nb_megabatches is None:
            raise Exception('User must specify the number of mega-batches in the stream')

        self.alma_random_seed = random_seed
        super().__init__(cl_dataset=cl_dataset, nb_tasks=nb_megabatches, \
                transformations=transformations, random_seed=random_seed)

    def _setup(self, nb_tasks: Optional[int]) -> int:
        x, y, t = self.cl_dataset.get_data()

        if len(y) < nb_tasks:
            raise Exception('Cannot have more tasks than available samples in the dataset')

        if t is not None:
            warnings.warn(
                f"The chosen dataset provides a task-id for each sample. This is ignored by ALMA."
            )

        if nb_tasks > 0:

            # unlike as in `InstanceIncremental`, we shuffle how samples are assigned to tasks.
            np.random.seed(self.alma_random_seed)
            idx = np.arange(len(y))
            idx = np.random.permutation(idx)
            chunks = np.array_split(idx, nb_tasks)
            task_ids = np.zeros_like(idx)
            for task, chunk in enumerate(chunks):
                task_ids[chunk] = task

            self.dataset = (x, y, task_ids)
        else:
            raise Exception(f"You must then specify a positive number of tasks, not {nb_tasks}.")

        return nb_tasks
