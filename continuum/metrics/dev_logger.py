import abc
import collections
import torch
import numpy as np
from continuum.metrics.base_logger import _BaseLogger
from continuum.metrics.metrics import get_model_size


class Dev_Logger(_BaseLogger):
    def __init__(self, root_log=".", list_keywords=["performance"], list_fields=["train", "eval"]):
        self.list_keywords = list_keywords
        self.list_fields = list_fields

        assert self.list_keywords is not None
        assert self.list_fields is not None
        assert len(self.list_keywords) >= 1
        assert len(self.list_subsets) >= 1

        logger_dic = {}

        for keyword in self.list_keywords:
            logger_dic[keyword] = {}
            for subset in self.list_subsets:
                logger_dic[keyword][subset] = {}

        self.current_task = 0
        self.current_epoch = 0

        self._predictions = collections.defaultdict(list)
        self._targets = collections.defaultdict(list)
        self._tasks = collections.defaultdict(list)
        self._model_sizes = []

        self._batch_predictions = []
        self._batch_targets = []

    def add(self, value, keyword, subset):

        assert keyword in self.list_keywords, f"Keyword {keyword} is not declared in list_keywords {self.list_keywords}"
        assert subset in self.list_subsets, f"Field {subset} is not declared in list_keywords {self.list_subsets}"

        if keyword == "performance":
            self.add_perf(value, subset)
        else:
            self.add_value(value, subset)

    def convert_numpy(self, _tensor):

        if isinstance(_tensor, torch.Tensor):
            _tensor = _tensor.cpu().numpy()
        return _tensor


    def add_perf(self, predictions=None, targets=None, task_ids=None, subset="train"):
        predictions = self.convert_numpy(predictions)
        targets = self.convert_numpy(targets)
        task_ids = self.convert_numpy(task_ids)

        if not isinstance(predictions, np.ndarray):
            raise TypeError(f"Provide predictions as np.array, not {type(predictions).__name__}.")
        if not isinstance(targets, np.ndarray):
            raise TypeError(f"Provide targets as np.array, not {type(predictions).__name__}.")

        self.logger_dic["performance"][subset][self.current_task][self.current_epoch]["predictions"].append(predictions)
        self.logger_dic["performance"][subset][self.current_task][self.current_epoch]["targets"].append(targets)
        self.logger_dic["performance"][subset][self.current_task][self.current_epoch]["task_ids"].append(task_ids)

    def new_epoch(self):
        self.current_epoch += 1

    def new_task(self):
        self.save_dic()
        self.current_task += 1
        self.epoch = 0

    def save_dic(self):
        import pickle as pkl
        filename = f"logger_dic_task_{self.current_task}.pkl"
        with open(filename, 'wb') as f:
            pkl.dump(self.logger_dic, f, pkl.HIGHEST_PROTOCOL)

        # after saving values we remove them from dictionnary to save space and memory
        for keyword in self.list_keywords:
            for subset in self.list_subsets:
                self.logger_dic["performance"][subset][self.current_task] = None
