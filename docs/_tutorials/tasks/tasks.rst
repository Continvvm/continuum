
In continual learning, tasks are empirically designed sets of data. Tasks have generally fixed data distribution or a fixed learning objective.
When the learning scenario jumps from one task to another it means that something has changed in the learning process, i.e. the data or the objective.

In Continuum the task_set class is a set of data that can be loaded with pytorch loader and sampled to feed data to an algorithm.
The scenarios are composed then by a sequence of task_set. Each task_set defines then a learning problem that the algorithm will have to solve one by one.

In Continuum, the user doesn't have to create tasks with task_set class, the tasks are created with the scenarios classes.
Thus we don't recommend you to play directly with this API if you're unsure about what you're doing.

Nevertheless, even if it is not very useful, it is still possible to create a task.

.. code-block:: python

    # create one task with all MNIST

    from torch.utils.data import DataLoader

    from continuum.datasets import MNIST
    from continuum import TaskSet

    dataset = MNIST("my/data/path", download=True, train=True)

    # get data x, label y and task index t
    x, y, t = dataset.get_data()
    transform = None

    # create the task set
    task = TaskSet(x, y, t, transform, data_type=dataset.data_type)

    # the task set can be use to create a loader
    task_loader = DataLoader(task)

    for x, y, t in task_loader:
        # sample batch of data to train your model
        # t is the task index or task label, if it is not defined t is a batch of -1
        break



Tasks are designed to conveniently split the training scenarios into clear learning problems.
In real-life scenarios, the separation of the learning process might be impossible, either because there is no clear transition or because we don't know when they happen, i.e. they are not labeled.


Useful methods
--------------------

- ``get_random_samples(nb_samples)`` provide random samples from the task, useful for visualization and control.

- ``get_raw_samples(indexes)`` provide samples from the task without applying any transformation, useful for visualization and control.

- ``get_classes()`` provide an array containing all the classes from the task_set object.

- ``nb_classes()`` provide the number of classes.

- ``add_samples(x: np.ndarray, y: np.ndarray, t: Union[None, np.ndarray] = None)`` makes possible to add manually data into the training set, might be useful for rehearsal strategies.

Useful functions
--------------------

- ``split_train_val(dataset, split_val = 0.1)`` split a task_set into two for validation purpose

.. code-block:: python

    task_set = MyTaskSet()

    # split the task_set such as 10% of the data, randomly selected, are used for validation.
    task_set_train, task_set_valid = (task_set, val_split = 0.1)
