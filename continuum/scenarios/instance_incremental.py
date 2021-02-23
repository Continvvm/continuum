import warnings
from typing import Callable, List, Optional

import numpy as np

from continuum.datasets import _ContinuumDataset
from continuum.scenarios import _BaseScenario


class InstanceIncremental(_BaseScenario):
    """Continual Loader, generating instance incremental consecutive tasks.

    Scenario: Classes are always the same but instances change (NI scenario)

    :param cl_dataset: A continual dataset.
    :param nb_tasks: The scenario number of tasks.
    :param transformations: List of transformations to apply to all tasks.
    :param random_seed: A random seed to init random processes.
    """

    def __init__(
        self,
        cl_dataset: _ContinuumDataset,
        nb_tasks: Optional[int] = None,
        transformations: List[Callable] = None,
        random_seed: int = 1
    ):
        super().__init__(cl_dataset=cl_dataset, nb_tasks=nb_tasks, transformations=transformations)

        self._random_state = np.random.RandomState(seed=random_seed)

        self._nb_tasks = self._setup(nb_tasks)

    def _setup(self, nb_tasks: Optional[int]) -> int:
        x, y, t = self.cl_dataset.get_data()

        if nb_tasks is not None and nb_tasks > 0:  # If the user wants a particular nb of tasks
            task_ids = _split_dataset(y, nb_tasks)
            self.dataset = (x, y, task_ids)
        elif t is not None:  # Otherwise use the default task ids if provided by the dataset
            self.dataset = (x, y, t)
            nb_tasks = len(np.unique(t))
        else:
            raise Exception(f"The dataset ({self.cl_dataset}) doesn't provide task ids, "
                            f"you must then specify a number of tasks, not ({nb_tasks}.")

        return nb_tasks


def _split_dataset(y, nb_tasks):
    nb_per_class = np.bincount(y)
    nb_per_class_per_task = nb_per_class / nb_tasks

    if (nb_per_class_per_task < 1.).all():
        raise Exception(f"Too many tasks ({nb_tasks}) for the amount of data "
                        "leading to empty tasks.")
    if (nb_per_class_per_task <= 1.).any():
        warnings.warn(
            f"Number of tasks ({nb_tasks}) is too big resulting in some tasks"
            " without all classes present."
        )

    n = nb_per_class_per_task.astype(np.int64)
    t = np.zeros((len(y),))

    for class_id, nb in enumerate(n):
        t_class = np.zeros((nb_per_class[class_id],))
        for task_id in range(nb_tasks):
            t_class[task_id * nb:(task_id + 1) * nb] = task_id

        t[y == class_id] = t_class

    return t
