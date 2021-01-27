
import collections
import torch
import numpy as np

from continuum.metrics.metrics import accuracy, \
    get_model_size_efficiency, \
    get_model_size, \
    forgetting, \
    accuracy_A, \
    remembering, \
    positive_backward_transfer, \
    forward_transfer, \
    backward_transfer

def require_subset(subset):
    def wrapper1(func):
        def wrapper2(self):
            if subset not in self._predictions:
                raise Exception(
                    f"No {subset} predictions have been logged so far which "
                    f"{func.__name__} rely on!"
                )
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


class Logger:
    def __init__(self):
        self._predictions = collections.defaultdict(list)
        self._targets = collections.defaultdict(list)
        self._tasks = collections.defaultdict(list)
        self._model_sizes = []

        self._batch_predictions = []
        self._batch_targets = []

    def add_batch(self, predictions, targets):
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.cpu().numpy()

        if not isinstance(predictions, np.ndarray):
            raise TypeError(f"Provide predictions as np.array, not {type(predictions).__name__}.")
        if not isinstance(targets, np.ndarray):
            raise TypeError(f"Provide targets as np.array, not {type(predictions).__name__}.")

        self._batch_predictions.append(predictions)
        self._batch_targets.append(targets)

    def add_step(self, predictions=None, targets=None, task_ids=None, subset="test", model=None):
        if subset not in ("train", "test"):
            raise ValueError(f"Subset must be train, val, or test, not {subset}.")

        if isinstance(predictions, torch.Tensor):
            predictions = predictions.cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.cpu().numpy()
        if isinstance(task_ids, torch.Tensor):
            task_ids = task_ids.cpu().numpy()

        if predictions is not None and targets is not None and task_ids is not None:
            self._predictions[subset].append(predictions)
            self._targets[subset].append(targets)
            self._tasks[subset].append(task_ids)

        if model is not None:
            self._model_sizes.append(get_model_size(model))

        # Remove all cached properties
        for k in list(self.__dict__.keys()):
            if k.startswith("__cached_"):
                del self.__dict__[k]

        self._batch_predictions = []
        self._batch_targets = []

    def log(self):
        print(f"Task id={self.nb_tasks}, acc={self.accuracy}, avg-acc={self.average_incremental_accuracy}")

    @property
    def nb_tasks(self):
        return len(self._predictions[list(self._predictions.keys())[0]])

    @property
    def online_accuracy(self):
        if len(self._batch_predictions) == 0:
            raise Exception(
                "You need to call <add_batch(preds, targets)> in order to get the online accuracy."
            )

        if len(self._batch_predictions) > 1:
            p, t = np.concatenate(self._batch_predictions), np.concatenate(self._batch_targets)
        else:
            p, t = self._batch_predictions[0], self._batch_targets[0]

        return accuracy(p, t)

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
    @require_subset("test")
    def accuracy_per_task(self):
        """Returns all task accuracy individually."""
        return [
            _get_R_ij(-1, j, all_preds, all_targets, all_tasks)
            for j in range(self.nb_tasks)
        ]

    @property
    @cache
    @require_subset("train")
    def online_cumulative_performance(self):
        """Computes the accuracy of last task on the train set.

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
        """Computes the average of the accuracies computed after each task.

        Reference:
        * iCaRL: Incremental Classifier and Representation Learning
          Rebuffi et al. CVPR 2017
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
    def forward_transfer(self):
        return forward_transfer(self._predictions["test"], self._targets["test"], self._tasks["test"])

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

    @property
    @cache
    @require_subset("test")
    def forgetting(self):
        return forgetting(self._predictions["test"], self._targets["test"], self._tasks["test"])

    @property
    @cache
    def model_size_efficiency(self):
        return get_model_size_efficiency(self._model_sizes)
