
import abc
import collections
import torch
import numpy as np

from continuum.metrics.metrics import get_model_size


class _BaseLogger(abc.ABC):
    def __init__(self):
        self._predictions = collections.defaultdict(list)
        self._targets = collections.defaultdict(list)
        self._tasks = collections.defaultdict(list)
        self._model_sizes = []

        self._batch_predictions = []
        self._batch_targets = []

        self.epoch = 0
        self.task_id = 0

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

    def new_epoch(self):
        self.epoch += 1

    def new_task(self):
        self.task_id += 1
        self.epoch = 0



