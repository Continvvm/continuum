import abc
import collections
import os
import torch
import numpy as np
from continuum.metrics.base_logger import _BaseLogger
from continuum.metrics.metrics import get_model_size


class Dev_Logger(_BaseLogger):
    def __init__(self, root_log=None, list_keywords=["performance"], list_subsets=["train", "eval"]):
        self.root_log = root_log
        self.list_keywords = list_keywords
        self.list_subsets = list_subsets

        assert self.list_keywords is not None
        assert self.list_fields is not None
        assert len(self.list_keywords) >= 1
        assert len(self.list_subsets) >= 1

        self.logger_dict = {}

        for keyword in self.list_keywords:
            self.logger_dict[keyword] = {}
            for subset in self.list_subsets:
                self.logger_dict[keyword][subset] = {}

        self.current_task = 0
        self.current_epoch = 0

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

        self.logger_dict["performance"][subset][self.current_task][self.current_epoch]["predictions"].append(predictions)
        self.logger_dict["performance"][subset][self.current_task][self.current_epoch]["targets"].append(targets)
        self.logger_dict["performance"][subset][self.current_task][self.current_epoch]["task_ids"].append(task_ids)

    def update_dict_architecture(self, update_task=False):
        for keyword in self.list_keywords:
            if update_task:
                self.logger_dict[keyword][subset][self.current_task] = {}
            for subset in self.list_subsets:
                self.logger_dict[keyword][subset][self.current_task][self.current_epoch] = {}

    def new_epoch(self):
        self.current_epoch += 1
        self.update_dict_architecture(update_task=False)

    def new_task(self):
        if self.root_log is not None:
            self.save_dic()
        self.current_task += 1
        self.current_epoch = 0
        self.update_dict_architecture(update_task=True)


    def save_dic(self):
        import pickle as pkl
        filename = f"logger_dic_task_{self.current_task}.pkl"
        filename = os.path.join(self.root_log, filename)
        with open(filename, 'wb') as f:
            pkl.dump(self.logger_dict, f, pkl.HIGHEST_PROTOCOL)

        # after saving values we remove them from dictionnary to save space and memory
        for keyword in self.list_keywords:
            for subset in self.list_subsets:
                self.logger_dict[keyword][subset][self.current_task] = None
