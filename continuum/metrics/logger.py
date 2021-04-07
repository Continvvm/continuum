import statistics

import numpy as np

from continuum.metrics.base_logger import _BaseLogger
from continuum.metrics.utils import require_subset, cache
from continuum.metrics.metrics import accuracy, \
    get_model_size_growth, \
    _get_R_ij, \
    forgetting, \
    accuracy_A, \
    remembering, \
    positive_backward_transfer, \
    forward_transfer, \
    backward_transfer


class Logger(_BaseLogger):
    def __init__(self, list_keywords=["performance"], list_subsets=["train", "test"], root_log=None):
        super().__init__(root_log=root_log, list_keywords=list_keywords, list_subsets=list_subsets)

    def log(self):
        print(f"Task id={self.nb_tasks}, acc={self.accuracy}, avg-acc={self.average_incremental_accuracy}")

    @property
    def nb_tasks(self):
        return self.current_task

    def _conv_list_vector(self, list_vector):
        if len(list_vector) > 1:
            vector = np.concatenate(list_vector)
        else:
            vector = list_vector[0]
        return vector

    def _get_best_epochs_perf(self, subset):
        """If there is no eval data, we assume that the best epoch for each task is the last one"""

        last_epoch_pred = []
        last_epoch_targets = []
        last_epoch_task_ids = []
        for task_id in range(len(self.logger_dict[subset]["performance"])):
            predictions = self.logger_dict[subset]["performance"][task_id][-1]["predictions"]
            targets = self.logger_dict[subset]["performance"][task_id][-1]["targets"]
            task_id = self.logger_dict[subset]["performance"][task_id][-1]["task_ids"]

            last_epoch_pred.append(predictions)
            last_epoch_targets.append(targets)
            last_epoch_task_ids.append(task_id)

        return last_epoch_pred, last_epoch_targets, last_epoch_task_ids

    def _get_best_epochs_data(self, keyword, subset):
        assert keyword != "performance", f"this method is not mode for performance keyword use _get_best_epochs_perf"
        list_values = []
        for task_id in range(self.current_task):
            list_values.append(self.logger_dict[subset][keyword][task_id][-1])
        return list_values

    def _get_best_epochs(self, keyword="performance", subset="train"):
        if keyword == "performance":
            values = self._get_best_epochs_perf(subset)
        else:
            values = self._get_best_epochs_data(keyword, subset)
        return values

    def get_logs(self, keyword, subset):
        return self.logger_dict[subset][keyword]

    @property
    @require_subset("train")
    def online_accuracy(self):
        if self._get_current_predictions("train").size == 0:
            raise Exception(
                "You need to call add([prediction, label, task_id]) in order to compute an online accuracy "
                "(add([prediction, label, None]) also works here, task_id is not needed)."
            )
        predictions = self._get_current_predictions("train")
        targets = self._get_current_targets("train")

        return accuracy(predictions, targets)

    @property
    @require_subset("test")
    def accuracy(self):
        return accuracy(
            self._get_current_predictions("test"),
            self._get_current_targets("test")
        )

    @property
    @require_subset("test")
    def accuracy_per_task(self):
        """Returns all task accuracy individually."""
        all_preds, all_targets, all_tasks = self._get_best_epochs(subset="test")
        return [
            _get_R_ij(-1, j, all_preds, all_targets, all_tasks)
            for j in range(self.nb_tasks)
        ]

    @property
    @require_subset("train")
    def online_cumulative_performance(self):
        """Computes the accuracy of last task on the train set.

        Reference:
        * Online Fast Adaptation and Knowledge Accumulation: a New Approach to Continual Learning
          Caccia et al. NeurIPS 2020
        """
        preds = np.concatenate(
            [dict_epoch['predictions'] for dict_epoch in self.logger_dict["train"]["performance"][self.current_task]]
        )
        targets = np.concatenate(
            [dict_epoch['targets'] for dict_epoch in self.logger_dict["train"]["performance"][self.current_task]]
        )
        return accuracy(preds, targets)

    @property
    @require_subset("test")
    def average_incremental_accuracy(self):
        """Computes the average of the accuracies computed after each task.

        Reference:
        * iCaRL: Incremental Classifier and Representation Learning
          Rebuffi et al. CVPR 2017
        """
        all_preds, all_targets, _ = self._get_best_epochs(subset="test")
        return statistics.mean([
            accuracy(all_preds[t], all_targets[t])
            for t in range(len(all_preds))
        ])

    @property
    @require_subset("test")
    def backward_transfer(self):
        all_preds, all_targets, task_ids = self._get_best_epochs(subset="test")
        return backward_transfer(all_preds, all_targets, task_ids)

    @property
    @require_subset("test")
    def forward_transfer(self):
        all_preds, all_targets, task_ids = self._get_best_epochs(subset="test")
        return forward_transfer(all_preds, all_targets, task_ids)

    @property
    @require_subset("test")
    def positive_backward_transfer(self):
        all_preds, all_targets, task_ids = self._get_best_epochs(subset="test")
        return positive_backward_transfer(all_preds, all_targets, task_ids)

    @property
    @require_subset("test")
    def remembering(self):
        all_preds, all_targets, task_ids = self._get_best_epochs(subset="test")
        return remembering(all_preds, all_targets, task_ids)

    @property
    @require_subset("test")
    def accuracy_A(self):
        all_preds, all_targets, task_ids = self._get_best_epochs(subset="test")
        return accuracy_A(all_preds, all_targets, task_ids)

    @property
    @require_subset("test")
    def forgetting(self):
        all_preds, all_targets, task_ids = self._get_best_epochs(subset="test")
        return forgetting(all_preds, all_targets, task_ids)

    @property
    def model_size_growth(self):
        assert "model_size" in self.list_keywords
        sizes = self._get_best_epochs("model_size")
        return get_model_size_growth(sizes)
