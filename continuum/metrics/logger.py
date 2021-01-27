import numpy as np

from continuum.metrics.base_logger import _BaseLogger
from continuum.metrics.utils import require_subset, cache
from continuum.metrics.metrics import accuracy, \
    get_model_size_efficiency, \
    forgetting, \
    accuracy_A, \
    remembering, \
    positive_backward_transfer, \
    forward_transfer, \
    backward_transfer


class Logger(_BaseLogger):
    def __init__(self):
        super().__init__()

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
