import warnings
from typing import Callable, List

import numpy as np

from clloader.datasets import _ContinuumDataset
from clloader.scenarios import _BaseCLLoader


class InstanceIncremental(_BaseCLLoader):
    """Continual Loader, generating datasets for the consecutive tasks.
    Scenario: Classes are always the same but instances change (NI scenario)

    :param cl_dataset: A continual dataset.
    :param increment: Either number of classes per task, or a list specifying for
                      every task the amount of new classes.
    :param initial_increment: A different task size applied only for the first task.
                              Desactivated if `increment` is a list.
    :param train_transformations: A list of data augmentation applied to the train set.
    :param common_transformations: A list of transformations applied to both the
                                   the train set and the test set. i.e. normalization,
                                   resizing, etc.
    :param evaluate_on: How to evaluate on val/test, either on all `seen` classes,
                        on the `current` classes, or on `all` classes.
    :param class_order: An optional custom class order, used for NC.
    """

    def __init__(
        self,
        cl_dataset: _ContinuumDataset,
        nb_tasks: int = 0,
        train_transformations: List[Callable] = None,
        common_transformations: List[Callable] = None,
        train: bool = True,
        random_seed: int = 1
    ):
        super().__init__(
            cl_dataset=cl_dataset,
            nb_tasks=nb_tasks,
            train_transformations=train_transformations,
            common_transformations=common_transformations,
            train=train
        )

        self._random_state = np.random.RandomState(seed=random_seed)

        self._nb_tasks = self._setup(nb_tasks)

    def _setup(self, nb_tasks: int) -> int:
        x, y, t = self.cl_dataset.init(train=self.train)

        if t is None and nb_tasks <= 0:
            raise ValueError(f"You need to specify a number of tasks > 0, not {nb_tasks}.")
        elif t is None:
            task_ids = self._random_state.randint(nb_tasks, size=len(y))
            self.dataset = (x, y, task_ids)
        else:
            default_nb_tasks = len(np.unique(t))
            if nb_tasks > 0 and default_nb_tasks > nb_tasks:
                warnings.warn(
                    f"The default number of task ({default_nb_tasks} is lower than"
                    f" the one asked ({nb_tasks}), some tasks will be removed."
                )
                indexes = np.where(t <= nb_tasks - 1)[0]
                x, y, t = x[indexes], y[indexes], t[indexes]
            elif nb_tasks > 0 and default_nb_tasks < nb_tasks:
                raise ValueError(
                    f"Cannot have {nb_tasks} tasks while this dataset"
                    f" at most {default_nb_tasks} tasks."
                )
            else:
                nb_tasks = default_nb_tasks

            self.dataset = (x, y, t)

        return nb_tasks
