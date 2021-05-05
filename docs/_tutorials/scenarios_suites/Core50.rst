CORe50
-----------------

Core50 is a dataset proposed in
`"CORe50: a new Dataset and Benchmark for Continuous Object Recognition" <http://proceedings.mlr.press/v78/lomonaco17a.html>`__.
This dataset proposed small videos of of 50 objects from 10 differents classes
with 11 background environment (more info in `core50 doc <https://vlomonaco.github.io/core50/index.html#dataset>`__ ).
This dataset was originally created to propose various continual learning settings.

Continuum Scenarios
##########
You can create automatically scenarios with continuum by setting the scenario and classification parameter in Core50 dataset.
It will provide different types of annotation for targets and tasks.
For classification, you can choose to use category (10 classes) annotation or object annotation (50 classes)>
For the task ids, you can choose among "classes", "domains" and "objects" how the task labels will be affected to data.

- Class Incremental
""""""""

We can create a simple class incremental setting.

.. code-block:: python

    from continuum.datasets import COre50
    # Same as :
    # dataset=Core50("/your/path", scenario="classes", classification="object")
    dataset=Core50("/your/path")
    # 5 tasks with 10 classes each
    scenario = ClassIncremental(dataset, nb_tasks=5)

- Instance Incremental
""""""""
-- Environment incremental Scenario --

.. code-block:: python

    from continuum.datasets import COre50
    dataset=Core50("/your/path", scenario="domains", classification="category")
    # 11 tasks with 10 classes each video in 1 environment each
    # classes are object class
    scenario = InstanceIncremental(dataset, nb_tasks=5)

-- Object incremental Scenario --
.. code-block:: python

    from continuum.datasets import COre50
    dataset=Core50("/your/path", scenario="objects", classification="object")
    # 50 tasks with 1 object videos in the 11 environments
    # classes are object ids (50 classes then)
    scenario = InstanceIncremental(dataset, nb_tasks=5)


- Classes and Instances Incremental
""""""""

Class and Instances Incremental scenarios are proposed in the scenario from the original paper (next section).


Original scenarios:
##########

CORe50 provides domain ids which are automatically picked up by the `InstanceIncremental` scenario:


.. code-block:: python

    from continuum import InstanceIncremental
    from continuum.datasets import Core50v2_79, Core50v2_196, Core50v2_391

    scenario_79 = InstanceIncremental(dataset=Core50v2_79("/my/path"))
    scenario_196 = InstanceIncremental(dataset=Core50v2_196("/my/path"))
    scenario_391 = InstanceIncremental(dataset=Core50v2_391("/my/path"))


The three available version of CORe50 have respectively 79, 196, and 391 tasks. Each task may bring
new classes AND new instances of past classes, akin to the `NIC scenario <http://proceedings.mlr.press/v78/lomonaco17a.html>`_.
