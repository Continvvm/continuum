import warnings
from typing import Callable, List, Optional, Union

from continuum.datasets import _ContinuumDataset
from continuum.scenarios import InstanceIncremental
from continuum.scenarios.instance_incremental import _split_dataset

import numpy as np

class ALMA(InstanceIncremental):
    """ALMA Loader, generating tasks by randomly partioning a fixed dataset.
    NOTE: this is analogous to InstanceIncremental, but reformatted with the
    language of the paper https://arxiv.org/abs/2106.09563

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

        if t is not None:
            warnings.warn(
                f"The chosen dataset provides a task-id for each sample. This is ignored by ALMA."
            )

        if nb_tasks is not None and nb_tasks > 0:  # If the user wants a particular nb of tasks
            task_ids = _split_dataset(y, nb_tasks)

            # unlike as in `InstanceIncremental`, we shuffle how samples are assigned to tasks.
            np.random.seed(self.alma_random_seed)
            permutation = np.random.permutation(np.arange(task_ids.shape[0]))
            task_ids = task_ids[permutation]

            self.dataset = (x, y, task_ids)
        else:
            raise Exception(f"The dataset ({self.cl_dataset}) doesn't provide task ids, "
                            f"you must then specify a number of tasks, not ({nb_tasks}.")

        return nb_tasks
