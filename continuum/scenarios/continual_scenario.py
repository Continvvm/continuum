import warnings
from copy import copy
from typing import Callable, List, Union

import numpy as np

from continuum.datasets import _ContinuumDataset
from continuum.scenarios import _BaseScenario


class ContinualScenario(_BaseScenario):
    """Continual Loader, generating datasets for the consecutive tasks.

    Scenario: the scenario is entirely defined by the task label vector in the cl_dataset

    :param cl_dataset: A continual dataset.
    :param transformations: A list of transformations applied to all tasks.
    """

    def __init__(
            self,
            cl_dataset: _ContinuumDataset,
            transformations: List[Callable] = None,
    ) -> None:
        self.check_data(cl_dataset)
        super().__init__(cl_dataset=cl_dataset, nb_tasks=self.nb_tasks,transformations=transformations)

    def check_data(self, cl_dataset: _ContinuumDataset):
        x, y, t = cl_dataset.get_data()

        assert t is not None, print("The t vector should be defined for this scenario")
        assert len(x) == len(y) == len(t), print("data, label and task label vectors need to have the same length")

        list_unique_tasks_ids = np.unique(t)
        self._nb_tasks = len(list_unique_tasks_ids)
        self.dataset = (x, y, t)

        # we order the list to check if all index from 0 to nb_tasks are in the list
        list_unique_tasks_ids.sort()

        assert np.all(list_unique_tasks_ids == np.arange(self.nb_tasks)), print(
            f"there should be at least one data point"
            f"for each task from task id equal"
            f"zero to num_tasks-1 \n (unique task indexes) {list_unique_tasks_ids} vs "
            f"(expected) {np.arange(self.nb_tasks)}")


    #nothing to do in the setup function
    def _setup(self, nb_tasks: int) -> int:
        return nb_tasks