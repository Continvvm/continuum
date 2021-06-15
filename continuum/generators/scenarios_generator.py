import abc

import torch
import numpy as np

from continuum.scenarios import _BaseScenario, create_subscenario, ClassIncremental


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

    def get_task_order(self, seed, nb_tasks: int = None):
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
        task_order = self.get_task_order(seed, nb_tasks)

        subscenario = create_subscenario(self.base_scenario, task_order[:nb_tasks])

        return subscenario


class ClassOrderGenerator(_BaseGenerator):
    """Class Order Generator, generate sub-scenario from a base scenario simply by changing class order.
    The difference with TaskOrderGenerator is that the classes inside a same task change. This class is only compatible
    with ClassIncremental scenarios.

        :param scenario: the base scenario to use to generate sub-scenarios
        """

    def __init__(
            self,
            scenario: ClassIncremental
    ) -> None:
        super().__init__(scenario)

        self.base_scenario = scenario
        self.list_classes = scenario.classes
        self.nb_generator = torch.Generator()
        self.task_order = None

    def get_class_order(self, seed):
        self.nb_generator.manual_seed(seed)
        # generate a random task order
        class_order = torch.randperm(len(self.list_classes), generator=self.nb_generator)
        return class_order


    def sample(self, seed: int = None) -> ClassIncremental:
        # seed the generator
        if seed is None:
            seed = np.random.randint(10000)

        # generate a random class order
        class_order = self.get_class_order(seed)
        new_list_class = self.list_classes[class_order]

        # We create a scenario from base_scenario
        initial_increment = self.base_scenario.initial_increment
        nb_tasks = self.base_scenario.nb_tasks
        increment = self.base_scenario.increment
        cl_dataset = self.base_scenario.cl_dataset
        transformations = self.base_scenario.transformations

        scenario = ClassIncremental(cl_dataset=cl_dataset,
                                    nb_tasks=nb_tasks,
                                    increment=increment,
                                    initial_increment=initial_increment,
                                    transformations=transformations,
                                    class_order=new_list_class)

        return scenario
