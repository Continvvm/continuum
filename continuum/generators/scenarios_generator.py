import abc

import torch
import numpy as np

from continuum.datasets import _ContinuumDataset
from continuum.scenarios import _BaseScenario, create_subscenario, ClassIncremental, HashedScenario


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
    def sample(self, seed: int = None, nb_tasks: int = None) -> int:
        """"method to sample a scenario from the generator."""
        raise NotImplementedError


class TaskOrderGenerator(_BaseGenerator):
    """Task Order Generator, generate sub-scenario from a base scenario simply by changing task order.

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
    The difference with TaskOrderGenerator is that the classes inside a same task change.
    This class is only compatible with ClassIncremental scenarios.

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

    def sample(self, seed: int = None, nb_tasks: int = None) -> _BaseScenario:

        if nb_tasks is None:
            nb_tasks = self.base_scenario.nb_tasks

        if nb_tasks != self.base_scenario.nb_tasks:
            AssertionError("You can not change the number of tasks in the generator")

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


class HashGenerator(_BaseGenerator):
    """Hash Generator, generate scenario from a set of parameters similar to scenario parameters and a list of hash name
    Hash generator will create a HashedScenario from a hash name randomly sampled in the list.

    :param cl_dataset: A continual dataset.
    :param list_hash: list of hash name
    :param nb_tasks: nb_tasks of each scenario, if None it will be automatically set
    :param transformations: A list of transformations applied to all tasks. If
                            it's a list of list, then the transformation will be
                            different per task.
    :param filename_hash_indexes: base name of a file to save scenarios indexes and reload them after
    :param split_task: Define if the task split will be automatic by clusterization of hashes or manually balanced
    """

    def __init__(
            self,
            cl_dataset: _ContinuumDataset,
            list_hash=None,
            nb_tasks=None,
            transformations=None,
            filename_hash_indexes=None,
            split_task="auto"
    ) -> None:
        self.cl_dataset = cl_dataset
        self.nb_tasks = nb_tasks
        self.transformations = transformations
        self.filename_hash_indexes = filename_hash_indexes
        self.split_task = split_task
        self._hash_name = None

        self.all_hashs = ["AverageHash", "Phash", "PhashSimple", "DhashH", "DhashV", "Whash", "ColorHash"
                          ] # , "CropResistantHash"

        # create default scenario to test parameters
        self.base_scenario = HashedScenario(cl_dataset=self.cl_dataset,
                                            hash_name="AverageHash",
                                            nb_tasks=nb_tasks,
                                            transformations=transformations,
                                            filename_hash_indexes=f"{filename_hash_indexes}_AverageHash",
                                            split_task=self.split_task)
        if list_hash is None:
            self.list_hash = self.all_hashs
        else:
            # all items of list_hash should be in self.all_hashs
            if not all(item in self.all_hashs for item in list_hash):
                AssertionError("Unknown hash name")
            else:
                self.list_hash = list_hash
        super().__init__(self.base_scenario)

    def get_rand_hash_name(self, seed):
        self.nb_generator.manual_seed(seed)
        # generate a random task order
        hash_id = torch.randperm(len(self.list_hash), generator=self.nb_generator)[0]
        return self.list_hash[hash_id]

    def sample(self, seed: int = None, nb_tasks: int = None) -> _BaseScenario:
        ''''create one scenario with a ramdomly sampled hash_name from self.list_hash'''

        if nb_tasks is None and self.nb_tasks is not None:
            nb_tasks = self.nb_tasks

        # seed the generator
        if seed is None:
            seed = np.random.randint(10000)

        # generate a random class order
        self._hash_name = self.get_rand_hash_name(seed)

        # We create a scenario from base_scenario
        scenario = HashedScenario(cl_dataset=self.cl_dataset,
                                  hash_name=self._hash_name,
                                  nb_tasks=nb_tasks,
                                  transformations=self.transformations,
                                  filename_hash_indexes=f"{self.filename_hash_indexes}_{self._hash_name}",
                                  split_task=self.split_task)

        return scenario

    @property
    def hash_name(self):
        ''''Hash name of the previous sampled scenario'''
        return self._hash_name
