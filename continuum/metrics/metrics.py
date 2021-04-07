from functools import reduce

import numpy as np


def accuracy(task_preds, task_targets):
    """Computes the accuracy of a given task.

    :param task_preds: Predicted labels.
    :param task_targets: Ground-truth targets.
    :return: a float metric between 0 and 1.
    """

    assert task_preds.size > 0
    assert task_targets.size > 0
    assert task_targets.size == task_preds.size, f"{task_targets.size} vs {task_preds.size}"

    metric = (task_preds == task_targets).mean()
    assert 0. <= metric <= 1.0, metric
    return metric


def accuracy_A(all_preds, all_targets, all_tasks):
    """Accuracy as defined in Diaz-Rodriguez and Lomonaco.

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

    :param all_preds: All predicted labels up to now.
    :param all_targets: All targets up to now.
    :param all_tasks: All task ids up to now.
    :return: a float metric between 0 and 1.
    """
    T = len(all_preds)  # Number of seen tasks so far
    # TODO if we take in account zeroshot, we should take the max of all_tasks?
    A = 0.

    for i in range(T):
        for j in range(i + 1):
            A += _get_R_ij(i, j, all_preds, all_targets, all_tasks)

    metric = A / (T * (T + 1) / 2)
    assert 0. <= metric <= 1.0, metric
    return metric


def backward_transfer(all_preds, all_targets, all_tasks):
    """Measures the influence that learning a task has on the performance on previous tasks.

    Reference:
    * Gradient Episodic Memory for Continual Learning
      Lopez-paz & ranzato, NeurIPS 2017

    :param all_preds: All predicted labels up to now.
    :param all_targets: All targets up to now.
    :param all_tasks: All task ids up to now.
    :return: a float metric between 0 and 1.
    """
    T = len(all_preds)  # Number of seen tasks so far
    # TODO if we take in account zeroshot, we should take the max of all_tasks?
    if T <= 1:
        return 0.
    bwt = 0.

    for i in range(1, T):
        for j in range(0, i - 1):
            r_ij = _get_R_ij(i, j, all_preds, all_targets, all_tasks)
            r_jj = _get_R_ij(j, j, all_preds, all_targets, all_tasks)

            bwt += (r_ij - r_jj)

    metric = bwt / (T * (T - 1) / 2)
    assert -1. <= metric <= 1.0, metric
    return metric


def positive_backward_transfer(all_preds, all_targets, all_tasks):
    """Computes the the positive gain of Backward transfer.

    Reference:
    * Don’t forget, there is more than forgetting: newmetrics for Continual Learning
      Diaz-Rodriguez and Lomonaco et al. NeurIPS Workshop 2018

    :param all_preds: All predicted labels up to now.
    :param all_targets: All targets up to now.
    :param all_tasks: All task ids up to now.
    :return: a float metric between 0 and 1.
    """
    bwt = backward_transfer(all_preds, all_targets, all_tasks)
    metric = 1 - abs(min(bwt, 0.))
    assert 0. <= metric <= 1.0, metric
    return metric


def remembering(all_preds, all_targets, all_tasks):
    """Computes the forgetting part of Backward transfer.

    Reference:
    * Don’t forget, there is more than forgetting: newmetrics for Continual Learning
      Diaz-Rodriguez and Lomonaco et al. NeurIPS Workshop 2018

    :param all_preds: All predicted labels up to now.
    :param all_targets: All targets up to now.
    :param all_tasks: All task ids up to now.
    :return: a float metric between 0 and 1.
    """
    bwt = backward_transfer(all_preds, all_targets, all_tasks)
    metric = max(bwt, 0.)
    assert 0. <= metric <= 1.0, metric
    return metric


def forward_transfer(all_preds, all_targets, all_tasks):
    """Measures the influence that learning a task has on the performance of future tasks.

    Reference:
    * Gradient Episodic Memory for Continual Learning
      Lopez-paz & ranzato, NeurIPS 2017

    :param all_preds: All predicted labels up to now.
    :param all_targets: All targets up to now.
    :param all_tasks: All task ids up to now.
    :return: a float metric between 0 and 1.
    """
    T = len(all_preds)  # Number of seen tasks so far
    # TODO if we take in account zeroshot, we should take the max of all_tasks?
    if T <= 1:
        return 0.

    fwt = 0.
    for i in range(T):
        for j in range(i):
            fwt += _get_R_ij(i, j, all_preds, all_targets, all_tasks)

    metric = fwt / (T * (T - 1) / 2)
    assert -1. <= metric <= 1.0, metric
    return metric


def forgetting(all_preds, all_targets, all_tasks):
    """Measures the average forgetting.

    Reference:
    * Riemannian Walk for Incremental Learning: Understanding Forgetting and Intransigence
      Chaudhry et al. ECCV 2018
    """
    T = len(all_preds)  # Number of seen tasks so far
    # TODO if we take in account zeroshot, we should take the max of all_tasks?
    if T <= 1:
        return 0.

    f = 0.
    for j in range(T - 1):
        r_kj = _get_R_ij(T - 1, j, all_preds, all_targets, all_tasks)
        r_lj = max(_get_R_ij(l, j, all_preds, all_targets, all_tasks) for l in range(T - 1))
        f += (r_lj - r_kj)

    metric = f / (T - 1)
    assert 0. <= metric <= 1.0, metric
    return metric


def _get_R_ij(i, j, all_preds, all_targets, all_tasks):
    """Computes an accuracy after task i on task j.

    R matrix:

          || T_e1 | T_e2 | T_e3
    ============================|
     T_r1 || R*  | R_ij  | R_ij |
    ----------------------------|
     T_r1 || R*  | R_ij  | R_ij |
    ----------------------------|
     T_r1 || R*  | R_ij  | R_ij |
    ============================|

    R_13 is the R of the first column and the third row.

    Reference:
    * Don’t forget, there is more than forgetting: newmetrics for Continual Learning
      Diaz-Rodriguez and Lomonaco et al. NeurIPS Workshop 2018

    :param i: Task id after which a model was trained.
    :param j: Task id of the test data.
    :param all_preds: All predicted labels up to now.
    :param all_targets: All targets up to now.
    :param all_tasks: All task ids up to now.
    :return: a float metric between 0 and 1.
    """
    preds = all_preds[i]
    targets = all_targets[i]
    tasks = all_tasks[i]

    indexes = tasks == j
    return (preds[indexes] == targets[indexes]).mean()


def get_model_size(model):
    """Computes the total number of parameters.

    :param model: A Pytorch's nn.Module model.
    :return: The number of parameters.
    """
    nb_params = 0
    # we want he number of parameter for inference
    model.eval()
    for w in model.parameters():
        if len(w.shape) > 0:  # Tensor
            nb_params += reduce(lambda a, b: a * b, w.shape)
        else:  # Scalar
            nb_params += 1

    return nb_params


def get_model_size_growth(model_sizes):
    """Computes the growth of the model sizes.

    Same as model size efficiency but with a less missleading name
    Reference:
    * Don’t forget, there is more than forgetting: newmetrics for Continual Learning
      Diaz-Rodriguez and Lomonaco et al. NeurIPS Workshop 2018

    :param model_sizes: A list of number of parameters, each being computed after a task.
    :return: a float metric between 0 and 1.
    """
    T = len(model_sizes)
    if T <= 1:
        return 1.0

    ms = 0.
    for i in range(T):
        ms += (model_sizes[0][0] / model_sizes[i][-1])

    metric = min(1., ms / T)
    assert 0. <= metric <= 1.0, metric
    return metric
