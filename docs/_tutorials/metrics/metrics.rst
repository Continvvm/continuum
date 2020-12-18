Metrics
-------

Continual Learning has many different metrics due to the nature of the task.
Continuum proposes a metric logger that accumulates the predictions of the model
and in return offers a large variety of metrics.

.. code-block:: python

	from torch.utils.data import DataLoader
    import numpy as np

    from continuum import MetricsLogger, ClassIncremental
    from continuum.datasets import MNIST

    train_scenario = ClassIncremental(
        MNIST(data_path="my/data/path", download=True, train=True),
        increment=2
     )
    test_scenario = ClassIncremental(
        MNIST(data_path="my/data/path", download=True, train=False),
        increment=2
     )

    model = ... Initialize your model here ...

    logger = MetricsLogger()

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

