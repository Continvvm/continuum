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


Detailed Example
-------

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

