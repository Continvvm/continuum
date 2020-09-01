Classic Scenarios
-----------------

We propose here a list of classic continual learning scenarios used in the literature. For each, scenarios we show how to create it. For using it, you may look at `scenarios documentation <https://continuum.readthedocs.io/en/latest/_tutorials/scenarios/scenarios.html>`_

- split MNIST: 5 tasks, number of classes per tasks:2
.. code-block:: python
    from continuum.datasets import MNIST
    from continuum import ClassIncremental
    continuum = ClassIncremental(
                                 MNIST(data_path="my/data/path", download=True, train=True),
                                 increment=2
     )

- split CIFAR100: 6 tasks, first 50 classes then 10 classes per tasks.
.. code-block:: python
    from continuum.datasets import CIFAR100
    from continuum import ClassIncremental
    continuum = ClassIncremental(
                                 CIFAR100(data_path="my/data/path", download=True, train=True),
                                 increment=10,
                                 initial_increment=50
      )

- Permut MNIST: 5 tasks with different label space for each task
.. code-block:: python
    from continuum import Permutations
    from continuum.datasets import MNIST

    dataset = MNIST(data_path="my/data/path", download=True, train=True)
    continuum = Permutations(
                             MNIST(data_path="my/data/path", download=True, train=True),
                             nb_tasks=5,
                             seed=0,
                             shared_label_space=False
    )

- Rotations MNIST: 3 tasks, rotation 0-45-90 degrees with different label space for each task
.. code-block:: python

    from continuum import Rotations
    from continuum.datasets import MNIST

    continuum = Rotations(
        MNIST(data_path="my/data/path", download=True, train=True),
        nb_tasks=3,
        list_degrees=[0,45,90]
    )


For more info `scenarios documentation <https://continuum.readthedocs.io/en/latest/_tutorials/scenarios/scenarios.html>`_.