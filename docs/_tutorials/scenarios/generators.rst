
Scenario Generators
-----------------

Scenario generators are object that can produce various version of a certain type of scenarios.
For example, they can generate scenario with various task orders.
It enables to test easily an algorithm on different flavors of a scenario types.
It also make possible to create a never ending sequence of tasks with repetition of task for long training.

Following the different types of generator:



Task Order Generator
----------------

This generator creates scenarios with various order of tasks:

.. code-block:: python

    from continuum.generators import TaskOrderGenerator

    # example with ClassIncremental scenario but any type of scenario works
    from continuum.scenarios import ClassIncremental
    from continuum.datasets import MNIST


    dataset = MNIST('my/data/path', train=True)
    scenario = ClassIncremental(dataset, increment=2)
    scenario_generator = TaskOrderGenerator(scenario)
    NB_SCENARIO = 5 # let say we want to generate 5 scenarios

    for scenario_id in range(NB_SCENARIO):
        # sample with seed for reproducibility
        scenario = scenario_generator.sample(seed=scenario_id)

        # each scenario has a different task order
        # (However, the tasks stays the same.)

Class Order Generator
----------------

This generator shuffle the class order but keeps the same number of classes per tasks.
This generator is suited for `ClassIncremental` scenarios.

.. code-block:: python

    from continuum.generators import ClassOrderGenerator
    from continuum.scenarios import ClassIncremental
    from continuum.datasets import MNIST


    dataset = MNIST('my/data/path', train=True)
    scenario = ClassIncremental(dataset, increment=[2, 3, 1, 4])
    scenario_generator = ClassOrderGenerator(scenario)
    NB_SCENARIO = 5 # let say we want to generate 5 scenarios

    for scenario_id in range(NB_SCENARIO):
        # sample with seed for reproducibility
        scenario = scenario_generator.sample(seed=scenario_id)

        # The increment order will stay [2, 3, 1, 4]
        # but the classes used in each task will be different