import warnings
from typing import Callable, List

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
        nb_tasks: int = 0,
        transformations: List[Callable] = None,
        random_seed: int = 1
    ):
        super().__init__(cl_dataset=cl_dataset, nb_tasks=nb_tasks, transformations=transformations)

        self._random_state = np.random.RandomState(seed=random_seed)

        self._nb_tasks = self._setup(nb_tasks)

    def _setup(self, nb_tasks: int) -> int:
        x, y, t = self.cl_dataset.get_data()

        if t is None and nb_tasks <= 0:
            raise ValueError(f"You need to specify a number of tasks > 0, not {nb_tasks}.")
        if t is None:  # If the dataset didn't provide default task ids:
            task_ids = _split_dataset(y, nb_tasks)
            self.dataset = (x, y, task_ids)
        else:  # With dataset default task ids provided:
            default_nb_tasks = len(np.unique(t))
            if default_nb_tasks > nb_tasks > 0:
                # If the user desired a particular amount of tasks, that is lower
                # than the dataset's default number of tasks, we truncate the
                # latest tasks.
                warnings.warn(
                    f"The default number of task ({default_nb_tasks} is lower than"
                    f" the one asked ({nb_tasks}), some tasks will be removed."
                )
                indexes = np.where(t <= nb_tasks - 1)[0]
                x, y, t = x[indexes], y[indexes], t[indexes]
            elif nb_tasks > 0 and default_nb_tasks < nb_tasks:
                # If the user requests more tasks than the dataset was designed for,
                # we raise an error.
                raise ValueError(
                    f"Cannot have {nb_tasks} tasks while this dataset"
                    f" at most {default_nb_tasks} tasks."
                )
            else:
                nb_tasks = default_nb_tasks

            self.dataset = (x, y, t)

        return nb_tasks



def _split_dataset(y, nb_tasks):
    nb_per_class = np.bincount(y)
    nb_per_class_per_task = nb_per_class / nb_tasks

    if (nb_per_class_per_task <= 0.).any():
        warnings.warn(
            f"Number of tasks ({nb_tasks}) is too big resulting in some tasks"
            " without all classes present."
        )

    n = nb_per_class_per_task.astype(np.int64)
    t = np.zeros((len(y),))

    for class_id, nb in enumerate(n):
        t_class = np.zeros((nb_per_class[class_id],))
        for task_id in range(nb_tasks):
            t_class[task_id * nb:(task_id + 1)* nb] = task_id

        t[y == class_id] = t_class

    return t
