
The continual learning (CL) aim is to learn without forgetting in the most "satisfying" way. To evaluate the capacity of different CL algorithms the community use different type of benchmarks scenarios. 

In Continuum, we implemented the four main types of scenarios used in the community: Class Incremental, Instance Incremental, New Class and Instance Incremental, and, Transformed Incremental. Those scenarios are originally suited for classification but they might be adapted for unsupervised learning, self-supervised learning or reinforcement learning. 

For each scenario, there is a finite set of tasks and the goal is to learn the tasks one by one and to be able to generalize at the end to a test set composed of data from all tasks.

For clear evaluation purpose, the data from each tasks can not be found in another tasks. The idea is to be able to assess precisely what the CL algorithms is able to remember or not. Those benchmarks purpose is not to mimic real-life scenarios but to evaluate properly algorithms capabilities.

Classes Incremental
--------------------

*In short:* 

Each new task bring instances from new classes only.

*Aim:* 

Evaluate the capability of an algorithms to learn concept sequentially, i.e. create representaion able to distinguish concepts and find the right decision boundaries without access to all past data.

*Some Details:*
 
The continuum of data is composed of several tasks. Each task contains class(es) that is/are specific to this task. One class can not be in several tasks.

One example, MNIST class incremental with five balanced tasks, MNIST has 10 classes then:
- task 0 contains data points labelled as 0 and 1
- task 1 contains data points labelled as 2 and 3
...
- task 4 contains data points labelled as 8 and 9

The Continual Loader `ClassIncremental` loads the data and batch it in several
tasks, each with new classes. See there some example arguments:

.. code-block:: python

    from continuum import ClassIncremental

    continuum = ClassIncremental(
        train_continual_dataset,
        increment=10,
        initial_increment=2,
        transformations=[
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ]
    )

Here the first task is made of 2 classes, then all following tasks of 10 classes. You can have a more finegrained increment by providing a list of `increment=[2, 10, 5, 10]`.
If you want a clloader for the test data, you'll need to create another one with a test continual dataset.

Instance Incremental
--------------------

*In short:* 

Each new tasks bring new instances from known classes.

*Aim:* 

Evaluate the capability of an algorithms to improve its generalization capabilities through new data points, i.e. improve representation without access to all past data.

*Some Details:*

Tasks are made of new instances. By default the samples images are randomly
shuffled in different tasks, but some datasets provide, in addition of the data `x` and labels `y`,
a task id `t` per sample. For example `MultiNLI`, a NLP dataset, has 5 classes but
with 10 different domains. Each domain represents a new task.


.. code-block:: python

    from continuum import InstanceIncremental
    from continuum.datasets import MultiNLI

    continuum = InstanceIncremental(
        MultiNLI("/my/path/where/to/download")
    )

Transformed Incremental
-----------------------

*In short:* Similar to instance incremental, each new tasks bring same instance with a different transformation (ex: images rotations, pixels permutations, ...)

*Aim:* Evaluate the capability of an algorithms to improve its generalization capabilities through new data points, i.e. improve representation without access to all past data.

*Some Details:*
The main difference with instance incremental, is that the scenarios builder has control of the different transformation spaces.
It is then easier to evaluate in which transformation space the algorithm is still able to generalize or not.

- Permutations Incremental [source](https://github.com/Continvvm/continuum/blob/master/continuum/scenarios/permutations.py)
is a famous case of TransformationIncremental class, in this case the transformation is a fixed pixel permutation. Each task has a specific permutation.
The scenarios is then to learn a same task in various permutation spaces.

- Rotations Incremental [source](https://github.com/Continvvm/continuum/blob/master/continuum/scenarios/rotations.py)
is also a famous case of TransformationIncremental class, in this case the transformation is a rotation of image. Each task has a specific rotation or range of rotation.
The scenarios is then to learn a same task in various rotations spaces.

New Class and Instance Incremental
----------------------------------

*In short:* Each new task bring both instances from new classes and new instances from known classes.

*Aim:* Evaluate the capability of an algorithms to both create new representation and improve existing ones.


*Some Details:*

NIC setting is a special case of NI setting. For now, only the CORe50 dataset
supports this setting.



Adding Your Own Scenarios
----------------------------------

Continuum is developed to be flexible and easily adapted to new settings.
Then you can create a new scenario by providing simply a new dataset framed in an existing scenatio such as Classes Incremental, Instance Incremental ...
You can also create a new class to create your own scenario with your own rules !

You can add it in the scenarios folder in the continuum project and make a pull request!

Scenarios can be seen as a list of [tasks](https://continuum.readthedocs.io/en/latest/_tutorials/datasets/tasks.html), the main thing to define is to define the content of each task to create a meaningful scenario.