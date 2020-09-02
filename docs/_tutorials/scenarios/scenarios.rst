Continuum Scenarios
-----------------

Continual learning (CL) aim is to learn without forgetting in the most "satisfying" way. To evaluate the capacity of different CL algorithms the community use different type of benchmarks scenarios.

In Continuum, we implemented the four main types of scenarios used in the community: Class Incremental, Instance Incremental, New Class and Instance Incremental, and, Transformed Incremental. Those scenarios are originally suited for classification but they might be adapted for unsupervised learning, self-supervised learning or reinforcement learning.

For each scenario, there is a finite set of tasks and the goal is to learn the tasks one by one and to be able to generalize at the end to a test set composed of data from all tasks.

For clear evaluation purpose, the data from each tasks can not be found in another tasks. The idea is to be able to assess precisely what the CL algorithms is able to remember or not. Those benchmarks purpose is not to mimic real-life scenarios but to evaluate properly algorithms capabilities.

Here is roughly how continual learning scenarios are created and used:


.. code-block:: python

	from torch.utils.data import DataLoader

    # First we get a dataset that will be used to compose tasks and the continuum
    continual_dataset = MyContinualDataset()

    # Then the dataset is provided to a scenario class that will process it to create the sequence of tasks
    # the continuum might get specific argument, such as number of tasks.
    scenario = MyContinuumScenario(
        continual_dataset, SomeOtherArguments
    )

    # The continuum can then be enumerate tasks
    for task_id, taskset in enumerate(scenario):
          # taskset can be used as a Pytorch Dataset to load the task data
          loader = DataLoader(taskset)

          for x, y, t in loader:
                # data, label, task index
                # train on the task here
                break


A practical example with split MNIST:

.. code-block:: python

    from torch.utils.data import DataLoader

    from continuum import ClassIncremental
    from continuum.datasets import MNIST
    from continuum.tasks import split_train_val


    dataset = MNIST(data_path="my/data/path", download=True, train=True)

    # split MNIST with 2 classes per tasks -> 5 tasks
    scenario = ClassIncremental(dataset, increment=2)

    # The continuum can then be enumerate tasks
    for task_id, taskset in enumerate(scenario):

        # We use here a cool function to split the dataset into train/val with 90% train
        train_taskset, val_taskset = split_train_val(taskset, val_split = 0.1)
        train_loader = DataLoader(train_taskset)

         # train dataset is a normal data loader like in pytorch that can be used to load the task data
        for x, y, t in train_loader:
            # data, label, task index
            # train on the task here
            break

    # For testing we need to create another loader (It is importan to keep test a train separate)
    dataset_test = MNIST(data_path="my/data/path", download=True, train=False)


    # You can also create a test scenario to frame test data as train data.
    scenario_test = ClassIncremental(dataset_test, increment=2)

    # then iterate through tasks
    for task_id, test_taskset in enumerate(scenario_test):
        test_loader = DataLoader(test_taskset)
        for x, y, t in test_loader:
            # something
            break

    # you can also select specific task(s) in the continuum
    # It's just python slicing!
    # select task i
    i = 2
    test_taskset = continuum_test[i]

    # select tasks i to i+2
    test_taskset = continuum_test[i:i+2]

    # select all seen tasks up to the i-th task
    test_taskset = continuum_test[:i + 1]

    # select all tasks
    test_taskset = continuum_test[:]


Classes Incremental
--------------------

**In short:**
Each new task bring instances from new classes only.

**Aim:**
Evaluate the capability of an algorithms to learn concept sequentially, i.e. create representaion able to distinguish concepts and find the right decision boundaries without access to all past data.

**Some Details:**
The continuum of data is composed of several tasks. Each task contains class(es) that is/are specific to this task. One class can not be in several tasks.

One example, MNIST class incremental with five balanced tasks, MNIST has 10 classes then:
- task 0 contains data points labelled as 0 and 1
- task 1 contains data points labelled as 2 and 3
...
- task 4 contains data points labelled as 8 and 9

The Continual Loader *ClassIncremental* loads the data and batch it in several
tasks, each with new classes. See there some example arguments:

.. code-block:: python

    from torchvision.transforms import transforms

    from continuum import ClassIncremental

    continual_dataset = MNIST(data_path="my/data/path", download=True, train=True)

    # first use case
    # first 2 classes per tasks
    scenario = ClassIncremental(
        continual_dataset,
        increment=2,
        transformations=[transforms.ToTensor()]
    )

    # second use case
    # first task with 2 classes then 4 classes per tasks until the end
    scenario = ClassIncremental(
        continual_dataset,
        increment=4,
        initial_increment=2,
        transformations=[transforms.ToTensor()]
    )

    # third use case
    # first task with 2, second task 3, third 1, ...
    scenario = ClassIncremental(
        continual_dataset,
        increment=[2, 3, 1, 4],
        transformations=[transforms.ToTensor()]
    )


Instance Incremental
--------------------

**In short:**
Each new tasks bring new instances from known classes.

**Aim:**
Evaluate the capability of an algorithms to improve its generalization capabilities through new data points, i.e. improve representation without access to all past data.

**Some Details:**
Tasks are made of new instances. By default the samples images are randomly
shuffled in different tasks, but some datasets provide, in addition of the data ``x`` and labels ``y``,
a task id ``t`` per sample. For example ``MultiNLI``, a NLP dataset, has 5 classes but
with 10 different domains. Each domain represents a new task.


.. code-block:: python

    from continuum import InstanceIncremental
    from continuum.datasets import MultiNLI

    dataset = MultiNLI("/my/path/where/to/download")
    scenario = InstanceIncremental(dataset=dataset)


Transformed Incremental
-----------------------

**In short:** Similar to instance incremental, each new tasks bring same instance with a different transformation (ex: images rotations, pixels permutations, ...)

**Aim:** Evaluate the capability of an algorithms to improve its generalization capabilities through new data points, i.e. improve representation without access to all past data.

**Some Details:**
The main difference with instance incremental, is that the scenarios builder has control of the different transformation spaces.
It is then easier to evaluate in which transformation space the algorithm is still able to generalize or not.

NB: the transformation used are `pytorch.transforms classes <https://pytorch.org/docs/stable/torchvision/transforms.html>`__

.. code-block:: python

    from continuum import TransformationIncremental

    list_of_transformation = [Trsf_0, Trsf_1, Trsf_2]

    # three tasks continuum, tasks 0 with Trsf_0 transformation
    scenario = TransformationIncremental(
        dataset=my_continual_dataset,
        incremental_transformations=list_of_transformation
    )



- Permutations Incremental `source <https://github.com/Continvvm/continuum/blob/master/continuum/scenarios/permutations.py>`__
is a famous case of TransformationIncremental class, in this case the transformation is a fixed pixel permutation. Each task has a specific permutation.
The scenarios is then to learn a same task in various permutation spaces.

.. code-block:: python

    from continuum import Permutations
    from continuum.datasets import MNIST

    dataset = MNIST(data_path="my/data/path", download=True, train=True)
    nb_tasks = 5
    seed = 0

    # A sequence of permutations is initialized from seed `seed` each task is with different pixel permutation
    # shared_label_space=True means that all classes use the same label space
    # ex: an image of the zeros digit will be always be labelized as a 0 ( if shared_label_space=False, zeros digit image permutated will got another label than the original one)
    scenario = Permutations(cl_dataset=dataset, nb_tasks=nb_tasks, seed=seed, shared_label_space=True)

- Rotations Incremental `source <https://github.com/Continvvm/continuum/blob/master/continuum/scenarios/rotations.py>`__
is also a famous case of TransformationIncremental class, in this case the transformation is a rotation of image. Each task has a specific rotation or range of rotation.
The scenarios is then to learn a same task in various rotations spaces.

.. code-block:: python

    from continuum import Rotations
    from continuum.datasets import MNIST

    nb_tasks = 3
    # first example with 3 tasks with fixed rotations
    list_degrees = [0, 45, 90]
    # second example with 3 tasks with ranges of rotations
    list_degrees = [0, (40,50), (85,95)]

    dataset = MNIST(data_path="my/data/path", download=True, train=True)
    scenario = Rotations(
        cl_dataset=dataset,
        nb_tasks=nb_tasks,
        list_degrees=list_degrees
    )


New Class and Instance Incremental
----------------------------------

**In short:** Each new task bring both instances from new classes and new instances from known classes.

**Aim:** Evaluate the capability of an algorithms to both create new representation and improve existing ones.

**Some Details:**
NIC setting is a special case of NI setting. For now, only the CORe50 dataset
supports this setting.

.. code-block:: python

    # Not implemented yet


Adding Your Own Scenarios
----------------------------------

Continuum is developed to be flexible and easily adapted to new settings.
Then you can create a new scenario by providing simply a new dataset framed in an existing scenario such as Classes Incremental, Instance Incremental ...
You can also create a new class to create your own scenario with your own rules !

You can add it in the scenarios folder in the continuum project and make a pull request!

Scenarios can be seen as a list of `tasks <https://continuum.readthedocs.io/en/latest/_tutorials/datasets/tasks.html>`__ , the main thing to define is to define the content of each task to create a meaningful scenario.
