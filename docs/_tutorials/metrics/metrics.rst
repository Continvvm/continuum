Metrics
-------

Continual Learning has many different metrics due to the nature of the task.
Continuum proposes a logger module that accumulates the predictions of the model.
Then the logger can compute various type of continual learning metrics based on the prediction saved.

Pseudo-code

.. code-block:: python

    for task in scenario:
        for (x,y,t) in tasks:
            predictions = model(x,y,t)

            logger.add_batch(predictions, y)

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
| **Model Size Efficiency**     | `model_size_efficiency`     |   ↓   |
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


**Model Size Efficiency**::

    Computes the efficiency of the model sizes.

    Reference:
    * Don’t forget, there is more than forgetting: newmetrics for Continual Learning
      Diaz-Rodriguez and Lomonaco et al. NeurIPS Workshop 2018



Detailed Example
----------------

.. code-block:: python

	from torch.utils.data import DataLoader
    import numpy as np

    from continuum import ClassIncremental
    from continuum.datasets import MNIST
    from continuum.metrics import Logger

    train_scenario = ClassIncremental(
        MNIST(data_path="my/data/path", download=True, train=True),
        increment=2
     )
    test_scenario = ClassIncremental(
        MNIST(data_path="my/data/path", download=True, train=False),
        increment=2
     )

    model = ... Initialize your model here ...

    logger = Logger()

    for task_id, (train_taskset, test_taskset) in enumerate(zip(train_scenario, test_scenario)):
        train_loader = DataLoader(train_taskset)
        test_loader = DataLoader(test_taskset)

        for x, y, t in train_loader:
            predictions = model(x)

            # Do here your model training with losses and optimizer...

            logger.add_batch(predictions, y)
            print(f"Online accuracy: {logger.only_accuracy}")

        preds, targets, task_ids = [], [], []
        for x, y, t in test_loader:
            preds.append(model(x).cpu().numpy())
            targets.append(y.cpu().numpy())
            task_ids.append(t.cpu().numpy())

        logger.add_step(
            np.concatenate(preds),
            np.concatenate(targets),
            np.concatenate(task_ids),
            model
        )
        print(f"Task: {task_id}, acc: {logger.accuracy}, avg acc: {logger.average_incremental_accuracy}")
        print(f"BWT: {logger.backward_transfer}, FWT: {logger.forward_transfer}")

