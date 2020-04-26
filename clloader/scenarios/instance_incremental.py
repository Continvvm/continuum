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
        nb_tasks: int,
        train_transformations: List[Callable] = None,
        common_transformations: List[Callable] = None,
        train=True
    ):
        super().__init__(
            cl_dataset=cl_dataset,
            nb_tasks=nb_tasks,
            train_transformations=train_transformations,
            common_transformations=common_transformations,
            train=train
        )

        self._nb_tasks = self._setup(nb_tasks)

    def _setup(self, nb_tasks: int) -> int:
        x, y, _ = self.cl_dataset.init(train=self.train)

        # TODO: need to add seed + randomstate to ensure reproducibility
        task_ids = np.random.randint(nb_tasks, size=len(y))

        self.dataset = (x, y, task_ids)

        return nb_tasks
