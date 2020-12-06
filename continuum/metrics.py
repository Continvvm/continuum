import statistics
import collections

import torch
import numpy as np


def require_subset(subset):
    def wrapper1(func):
        def wrapper2(self):
            if subset not in self._predictions:
                raise Exception(f"No {subset} predictions have been logged so far which last_accuracy rely on!")
            return func(self)
        return wrapper2
    return wrapper1


def cache(func):
    def wrapper(self):
        name = f"__cached_{func.__name__}"
        v = self.__dict__.get(name)
        if v is None:
            v = func(self)
            self.__dict__[name] = v
        return v
    return wrapper


class MetricsLogger:
    def __init__(self):
        self._predictions = collections.defaultdict(list)
        self._targets = collections.defaultdict(list)
        self._tasks = collections.defaultdict(list)

        self._unique_tasks = set()

    def add(self, predictions, targets, task_ids, subset="test"):
        if subset not in ("train", "val", "test"):
            raise ValueError(f"Subset must be train, val, or test, not {subset}.")

        self._predictions[subset].append(predictions)
        self._targets[subset].append(targets)
        self._tasks[subset].append(task_ids)

        # Remove all cached properties
        for k in list(self.__dict__.keys()):
            if k.startswith("__cached_"):
                del self.__dict__[k]

    @property
    @cache
    @require_subset("test")
    def accuracy(self):
        return accuracy(
            self._predictions["test"][-1],
            self._targets["test"][-1]
        )

    @property
    @cache
    @require_subset("train")
    def online_cumulative_performance(self):
        """TODO

        Reference:
        * Online Fast Adaptation and Knowledge Accumulation: a New Approach to Continual Learning
          Caccia et al. NeurIPS 2020
        """
        return accuracy(
            self._predictions["train"][-1],
            self._targets["train"][-1]
        )

    @property
    @cache
    @require_subset("test")
    def average_incremental_accuracy(self):
        """TODO

        Reference:
        * Rebuffi icarl
        """
        return statistics.mean([
            accuracy(self._predictions["test"][t], self._targets["test"][t])
            for t in range(len(self._predictions["test"]))
        ])

    @property
    @cache
    @require_subset("test")
    def backward_transfer(self):
        return backward_transfer(self._predictions["test"], self._targets["test"], self._tasks["test"])

    @property
    @cache
    @require_subset("test")
    def positive_backward_transfer(self):
        return positive_backward_transfer(self._predictions["test"], self._targets["test"], self._tasks["test"])

    @property
    @cache
    @require_subset("test")
    def remembering(self):
        return remembering(self._predictions["test"], self._targets["test"], self._tasks["test"])

    @property
    @cache
    @require_subset("test")
    def accuracy_A(self):
        return accuracy_A(self._predictions["test"], self._targets["test"], self._tasks["test"])


def accuracy(task_preds, task_targets):
    """Compute the accuracy of a given task.

    :param task_preds: Predicted labels.
    :param task_targets: Ground-truth targets.
    :return: a float metric between 0 and 1.
    """
    return (task_preds == task_targets).mean()


def accuracy_A(all_preds, all_targets, all_tasks):
    """Accuracy as defined in Diaz-Rodriguez and Lomonaco.

    Reference:
    * Don’t forget, there is more than forgetting: newmetrics for Continual Learning
      Diaz-Rodriguez and Lomonaco et al. NeurIPS Workshop 2018

    :param all_preds: All predicted labels up to now.
    :param all_targets: All targets up to now.
    :param all_tasks: All task ids up to now.
    :return: a float metric between 0 and 1.
    """
    T = len(all_preds)  # Number of seen tasks so far
    # TODO if we take in account zeroshot, we should take the max of all_tasks
    A = 0.

    for i in range(T):
        for j in range(i+1):
            A += _get_R_ij(i, j, all_preds, all_targets, all_tasks)

    return A / (T * (T + 1) / 2)


def backward_transfer(all_preds, all_targets, all_tasks):
    """TODO

    Reference:
    * Lopez-paz & ranzato 2017
    """
    T = len(all_preds)  # Number of seen tasks so far
    # TODO if we take in account zeroshot, we should take the max of all_tasks
    if T == 1:
        return 0.
    bwt = 0.

    for i in range(1, T):
        for j in range(0, i - 1):
            r_ij = _get_R_ij(i, j, all_preds, all_targets, all_tasks)
            r_jj = _get_R_ij(j, j, all_preds, all_targets, all_tasks)

            bwt += (r_ij - r_jj)

    return bwt / (T * (T - 1) / 2)


def positive_backward_transfer(all_preds, all_targets, all_tasks):
    """TODO

    Reference:
    * Don’t forget, there is more than forgetting: newmetrics for Continual Learning
      Diaz-Rodriguez and Lomonaco et al. NeurIPS Workshop 2018

    :param all_preds: All predicted labels up to now.
    :param all_targets: All targets up to now.
    :param all_tasks: All task ids up to now.
    :return: a float metric between 0 and 1.
    """
    bwt = backward_transfer(all_preds, all_targets, all_tasks)
    return 1 - abs(min(bwt, 0.))


def remembering(all_preds, all_targets, all_tasks):
    """TODO

    Reference:
    * Don’t forget, there is more than forgetting: newmetrics for Continual Learning
      Diaz-Rodriguez and Lomonaco et al. NeurIPS Workshop 2018

    :param all_preds: All predicted labels up to now.
    :param all_targets: All targets up to now.
    :param all_tasks: All task ids up to now.
    :return: a float metric between 0 and 1.
    """
    bwt = backward_transfer(all_preds, all_targets, all_tasks)
    return max(bwt, 0.)


def forward_transfer(all_preds, all_targets, all_tasks):
    """TODO

    Reference:
    * Lopez-paz & ranzato 2017
    """
    T = len(all_preds)  # Number of seen tasks so far
    # TODO if we take in account zeroshot, we should take the max of all_tasks


def _get_R_ij(i, j, all_preds, all_targets, all_tasks):
    """TODO

    Reference:
    * Don’t forget, there is more than forgetting: newmetrics for Continual Learning
      Diaz-Rodriguez and Lomonaco et al. NeurIPS Workshop 2018

    :param i: TODO
    :param j: TODO
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
