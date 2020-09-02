Classic Scenarios
-----------------

We propose here a list of classic continual learning scenarios used in the literature. For each, scenarios we show how to create it. For using it, you may look at `scenarios documentation <https://continuum.readthedocs.io/en/latest/_tutorials/scenarios/scenarios.html>`_

- split MNIST: 5 tasks, number of classes per tasks: 2

.. code-block:: python

    from continuum import ClassIncremental
    from continuum.datasets import MNIST

    scenario = ClassIncremental(
        MNIST(data_path="my/data/path", download=True, train=True),
        increment=2
     )


- split CIFAR100: 6 tasks, first 50 classes then 10 classes per tasks.

.. code-block:: python

    from continuum import ClassIncremental
    from continuum.datasets import CIFAR100

    scenario = ClassIncremental(
        CIFAR100(data_path="my/data/path", download=True, train=True),
        increment=10,
        initial_increment=50
      )


- Concatenation of CIFAR10 & CIFAR100, made of 11 tasks of 10 classes each

.. code-block:: python

    from continuum import ClassIncremental
    from continuum.datasets import CIFARFellowship

    scenario = ClassIncremental(
        CIFARFellowship(data_path="my/data/path", download=True, train=True),
        increment=10,
    )


- Permut MNIST: 5 tasks with different label space for each task

.. code-block:: python

    from continuum import Permutations
    from continuum.datasets import MNIST

    scenario = Permutations(
        MNIST(data_path="my/data/path", download=True, train=True),
        nb_tasks=5,
        seed=0,
        shared_label_space=False
    )

- Rotations MNIST: 3 tasks, rotation 0-45-90 degrees with different label space for each task

.. code-block:: python

    from continuum import Rotations
    from continuum.datasets import MNIST
    scenario = Rotations(
        MNIST(data_path="my/data/path", download=True, train=True),
        nb_tasks=3,
        list_degrees=[0,45,90]
    )


For more info `scenarios documentation <https://continuum.readthedocs.io/en/latest/_tutorials/scenarios/scenarios.html>`__.
