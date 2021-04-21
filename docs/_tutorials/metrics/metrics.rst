Metrics
-------

Continual Learning has many different metrics due to the nature of the task.
Continuum proposes a logger module that accumulates the predictions of the model.
Then the logger can compute various type of continual learning metrics based on the prediction saved.

Pseudo-code

.. code-block:: python

    logger = Logger()
    for task in scenario:
        for (x,y,t) in tasks:
            predictions = model(x,y,t)

            logger.add([predictions, y, t])
        logger.end_task()
    print(f"Metric result: {logger.my_pretty_metric}")

Here is a list of all implemented metrics:

+-------------------------------+-----------------------------+-------+
|Name                           | Code                        | ↑ / ↓ |
+===============================+=============================+=======+
| **Accuracy**                  | `accuracy`                  |   ↑   |
+-------------------------------+-----------------------------+-------+
| **Accuracy A**                | `accuracy_A`                |   ↑   |
+-------------------------------+-----------------------------+-------+
| **Backward Transfer**         | `backward_transfer`         |   ↑   |
+-------------------------------+-----------------------------+-------+
| **Positive Backward Transfer**| `positive_backward_transfer`|   ↑   |
+-------------------------------+-----------------------------+-------+
| **Remembering**               | `remembering`               |   ↑   |
+-------------------------------+-----------------------------+-------+
| **Forward Transfer**          | `forward_transfer`          |   ↑   |
+-------------------------------+-----------------------------+-------+
| **Forgetting**                | `forgetting`                |   ↓   |
+-------------------------------+-----------------------------+-------+
| **Model Size Growth**         | `model_size_growth`         |   ↓   |
+-------------------------------+-----------------------------+-------+

**Accuracy**::

    Computes the accuracy of a given task.


**Accuracy A**::

    Accuracy as defined in Diaz-Rodriguez and Lomonaco.

    Note that it is slightly different from the normal accuracy as it considers
    each task accuracy with equal weight, while the normal accuracy considers
    the proportion of all targets.

    Example:
    - Given task 1 with 50,000 images and task 2 with 1,000 images.
    - With normal accuracy, task 1 has more importance in the average accuracy.
    - With this accuracy A, task 1 has as much importance as task 2.

    Reference:
    * Don’t forget, there is more than forgetting: newmetrics for Continual Learning
      Diaz-Rodriguez and Lomonaco et al. NeurIPS Workshop 2018


**Backward Transfer**::

    Measures the influence that learning a task has on the performance on previous tasks.

    Reference:
    * Gradient Episodic Memory for Continual Learning
      Lopez-paz & ranzato, NeurIPS 2017


**Positive Backward Transfer**::

    Computes the the positive gain of Backward transfer.

    Reference:
    * Don’t forget, there is more than forgetting: newmetrics for Continual Learning
      Diaz-Rodriguez and Lomonaco et al. NeurIPS Workshop 2018


**Remembering**::

    Computes the forgetting part of Backward transfer.

    Reference:
    * Don’t forget, there is more than forgetting: newmetrics for Continual Learning
      Diaz-Rodriguez and Lomonaco et al. NeurIPS Workshop 2018


**Forward Transfer**::

    Measures the influence that learning a task has on the performance of future tasks.

    Reference:
    * Gradient Episodic Memory for Continual Learning
      Lopez-paz & ranzato, NeurIPS 2017


**Forgetting**::

    Measures the average forgetting.

    Reference:
    * Riemannian Walk for Incremental Learning: Understanding Forgetting and Intransigence
      Chaudhry et al. ECCV 2018


**Model Size Growth**::

    Evaluate the evolution of the model size.


Detailed Example
----------------

.. code-block:: python

    from torch.utils.data import DataLoader
    import numpy as np

    from continuum import ClassIncremental
    from continuum.datasets import MNIST
    from continuum.metrics import Logger

    train_scenario = ClassIncremental(
        MNIST(data_path='my/data/path', download=True, train=True),
        increment=2
     )

    test_scenario = ClassIncremental(
        MNIST(data_path='my/data/path', download=True, train=False),
        increment=2
     )

    model = a_model() #... Initialize your model here ...

    logger = Logger(list_subsets=['train', 'test'])

    for task_id, train_taskset in enumerate(train_scenario):
        train_loader = DataLoader(train_taskset, batch_size=len(train_taskset))
        test_taskset = test_scenario[:task_id + 1]  # Evaluating on all seen tasks
        test_loader = DataLoader(test_taskset, batch_size=len(test_taskset))

        for x, y, t in train_loader:
            predictions = model(x)

            # Do here your model training with losses and optimizer...

            logger.add([predictions, y, t], subset='train')
            print(f"Online accuracy: {logger.online_accuracy}")

        for x, y, t in test_loader:
            pred = model(x, t)
            logger.add([pred, y, t], subset='test')

        print(f"Task: {task_id}, acc: {logger.accuracy}, avg acc: {logger.average_incremental_accuracy}")
        print(f"BWT: {logger.backward_transfer}, FWT: {logger.forward_transfer}")
        logger.end_task()



Advanced Use of logger
--------------------------

The logger is designed to save any type of tensor with a corresponding keyword.
For example you may want to save a latent vector at each epoch.

.. code-block:: python

    from continuum.metrics import Logger

    model = ... Initialize your model here ...

    list_keywords=["latent_vector"]

    logger = Logger(list_keywords=list_keywords, list_subsets=['train', 'test'])

    for tasks in task_scenario):
        for epoch in range(epochs)
            for x, y, t in task_loader:
                # Do here your model training with losses and optimizer...
            latent_vector = model.get_latent_vector_fancy_method_you_designed()
            logger.add(latent_vector, keyword='latent_vector', subset="train")
            logger.end_epoch()

        logger.end_task()


If you want to log result to compute metrics AND log you latent vector you can declare and use you logger as following:

.. code-block:: python

    # Logger declaration with several keyword
    logger = Logger(list_keywords=["performance", "latent_vector"], list_subsets=['train', 'test'])

    # [...]
    # log test results for metrics
    logger.add([x,y,t], keyword='performance', subset="test")

    # [...]
    # log latent vector while testing
    logger.add(latent_vector, keyword='latent_vector', subset="test")

At the end of training or when you want, you can get all the data logged.

.. code-block:: python

    logger = Logger(list_keywords=["performance", "latent_vector"], list_subsets=['train', 'test'])

    # [... a long training a logging adventure ... ]

    logs_latent = logger.get_logs(keyword='latent_vector', subset='test')

    # you can explore the logs as follow
    for task_id in range(logs_latent):
        for epoch_id in range(logs_latent[task_id]):
            # the list of all latent vector you saved as task_id and epoch_id by chronological order.
            list_of_latent_vector_logged = logs_latent[task_id][epoch_id]

We hope it might be useful for you :)
