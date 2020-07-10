Welcome to Continuum's documentation!
=====================================

**Continuum** is the library you need for Continual Learning. It supports many
datasets and most scenarios (NC, NI, NIC, etc.).

Continuum is made of two parts: **Dataset** and **Scenario**. The former is about
the actual dataset with sometimes minor modifications to fit in the Continual paradigm.
The latter is about the different setting you may encounter in Continual Learning.

.. toctree::
   :glob:
   :maxdepth: 2
   :caption: Scenarios:

   _tutorials/scenarios/*

.. toctree::
   :glob:
   :maxdepth: 2
   :caption: Datasets:

   _tutorials/datasets/*


Quick Example
-------------

Let's have a quick look with an example. We want to evaluate our model on a split-MNIST.
First, we initialize the dataset; if the data isn't present at the given path, we'll
download it for you! All datasets but ImageNet have this feature.

Then we specify our dataset, here it is :code:`ClassIncremental` (*New Classes* NC) where each task
brings new classes. Our first task is made of 5 classes (:code:`initial_increment=5`),
then all 5 following tasks will be made of a single new class each:

.. code-block:: python

   from torch.utils.data import DataLoader

   from continuum import ClassIncremental, split_train_val
   from continuum.datasets import MNIST

   dataset = MNIST("my/data/path", download=True)

   clloader = ClassIncremental(
      dataset,
      increment=1,
      initial_increment=5,
      train=True
   )

   print(f"Number of classes: {clloader.nb_classes}.")
   print(f"Number of tasks: {clloader.nb_tasks}.")

   for task_id, train_dataset in enumerate(clloader):
      train_dataset, val_dataset = split_train_val(train_dataset, val_split=0.1)
      train_loader = DataLoader(train_dataset)
      val_loader = DataLoader(val_dataset)

      # Do your cool stuff here


The continual loader, here named :code:`clloader` is an iterable. Each loop provides then
the dataset for a task. We split the dataset into a train and validation subset with
our utility :code:`split_train_val`.

Note that if we wanted to use the test subset, we need to specify it when
creating the scenario.

Finally, after training on the 6 tasks, we want to evaluate our model performance
on the test set:

.. code-block:: python
   clloader_test = ClassIncremental(
      dataset,
      increment=1,
      initial_increment=5,
      train=False
   )

From this loader, we can get the first task :code:`clloader_test[0]`, all tasks up to the third
task :code:`clloader_test[:3]`, or even all tasks :code:`clloader_test[:]`. You can slice
any loaders like you would do with Python's List.


.. toctree::
   :glob:
   :maxdepth: 2
   :caption: Code Documentation:

   _autogen/continuum*

This library was developped by `Arthur Douillard <https://arthurdouillard.com/>`
and `Timothée Lesort <https://tlesort.github.io/>`. If you have any new feature
request or want to report a bug, please open an issue `here <https://github.com/Continvvm/continuum/issues>`.
We are also open to `pull requests <https://github.com/Continvvm/continuum/pulls>`!
