
The continual learning (CL) aim is to learn without forgetting in the most "satisfying" way. To evaluate the capacity of different CL algorithms the community use different type of benchmarks scenarios. 

In Continuum, we implemented the four main types of scenarios used in the community: Class Incremental, Instance Incremental, New Class and Instance Incremental, and, Transformed Incremental. Those scenarios are originally suited for classification but they might be adapted for unsupervised learning, self-supervised learning or reinforcement learning. 

For each scenario, there is a finite set of tasks and the goal is to learn the tasks one by one and to be able to generalize at the end to a test set composed of data from all tasks.

For clear evalutaion purpose, the data from each tasks can not be found in another tasks. The idea is to be able to assess precisely what the CL algorithms is able to remember or not. Those benchmarks purpose is not to mimic real-life scenarios but to evaluate properly algorithms capabilities.

Class Incremental
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

Instance Incremental
--------------------

*In short:* Each new tasks bring new instances from known classes.

*Aim:* Evaluate the capability of an algorithms to improve its generalization capabilities through new data points, i.e. improve representation without access to all past data.

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



Adding Your Own Scenarios
----------------------------------

