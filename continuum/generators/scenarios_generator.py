import abc

import torch
import numpy as np

from continuum.scenarios import _BaseScenario, create_subscenario


class _BaseGenerator(abc.ABC):
    """Abstract loader.

    DO NOT INSTANTIATE THIS CLASS.

    :param scenario: A Continuum scenarios
    """

    def __init__(
            self,
            scenario: _BaseScenario
    ) -> None:
        self.base_scenario = scenario
        self.nb_generator = torch.Generator()

    @abc.abstractmethod
    def sample(self, seed: int) -> int:
        raise NotImplementedError


class TaskOrderGenerator(_BaseGenerator):
    """Task Order Generator, generate sub-scenario from a base scenario simply by changing task order

        :param scenario: the base scenario to use to generate sub-scenarios
        """

    def __init__(
            self,
            scenario: _BaseScenario
    ) -> None:
        super().__init__(scenario)

        self.base_scenario = scenario
        self.nb_generator = torch.Generator()
        self.task_order = None

    def get_class_order(self, seed, nb_tasks: int = None):
        self.nb_generator.manual_seed(seed)

        # generate a random task order
        task_order = torch.randperm(self.base_scenario.nb_tasks, generator=self.nb_generator)
        if nb_tasks is None:
            nb_tasks = self.base_scenario.nb_tasks
        return task_order[:nb_tasks]

    def sample(self, seed: int = None, nb_tasks: int = None) -> _BaseScenario:
        # seed the generator
        if seed is None:
            seed = np.random.randint(10000)

        # generate a random task order
        task_order = self.get_class_order(seed, nb_tasks)

        subscenario = create_subscenario(self.base_scenario, task_order[:nb_tasks])

        return subscenario
