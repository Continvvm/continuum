
The continual learning (CL) aim is to learn without forgetting in the most "satisfying" way. To evaluate the capacity of different CL algorithms the community use different type of benchmarks scenarios. 

In Continuum, we implemented the four main types of scenarios used in the community: Class Incremental, Instance Incremental, New Class and Instance Incremental, and, Transformed Incremental. Those scenarios are originally suited for classification but they might be adapted for unsupervised learning, self-supervised learning or reinforcement learning. 

For each scenario, there is a finite set of tasks and the goal is to learn the tasks one by one and to be able to generalize at the end to a test set composed of data from all tasks.

For clear evalutaion purpose, the data from each tasks can not be found in another tasks. The idea is to be able to assess precisely what the CL algorithms is able to remember or not. Those benchmarks purpose is not to mimic real-life scenarios but to evaluate properly algorithms capabilities.

Classes Incremental
-----------------

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

```python
from continuum import ClassIncremental

clloader = ClassIncremental(
    my_continual_dataset,
    increment=10,
    initial_increment=2,
    train_transformations=[transforms.RandomHorizontalFlip()],
    common_transformations=[
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ],
    train=True
)
```

Here the first task is made of 2 classes, then all following tasks of 10 classes. You can have a more finegrained increment by providing a list of `increment=[2, 10, 5, 10]`.

The `train_transformations` is applied only on the training data, while the `common_transformations` on both the training and testing data.

If you want a clloader for the test data, you'll need to create a new instance with `train=False`.

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

```python
from continuum import InstanceIncremental
from continuum.datasets import MultiNLI

clloader = InstanceIncremental(
    MultiNLI("/my/path/where/to/download"),
    train=True
)
```

Transformed Incremental
-----------------------

*In short:* Similar to instance incremental, each new tasks bring same instance with a different transformation (ex: images rotations, pixels permutations, ...)

*Aim:* Evaluate the capability of an algorithms to improve its generalization capabilities through new data points, i.e. improve representation without access to all past data.

*Some Details:*
The main difference with instance incremental, is that the scenarios builder has control of the different transformation spaces. It is then easier to evaluate in which transformation space the algorithm is still able to generalize or not.

New Class and Instance Incremental
----------------------------------

*In short:* Each new task bring both instances from new classes and new instances from known classes.

*Aim:* Evaluate the capability of an algorithms to both create new representation and improve existing ones.


*Some Details:*

NIC settting is a special case of NI setting. For now, only the CORe50 dataset
supports this setting.



Adding Your Own Scenarios
----------------------------------

