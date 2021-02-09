import abc
import collections
import os
import torch
import numpy as np
from copy import deepcopy
from continuum.metrics.metrics import get_model_size


def convert_numpy(_tensor):
    if isinstance(_tensor, torch.Tensor):
        _tensor = _tensor.cpu().numpy()
    return _tensor


class _BaseLogger(abc.ABC):
    def __init__(self, root_log=None, list_keywords=["performance"], list_subsets=["train", "test"]):
        """
        root_log: folder where logged informations will be saved
        list_keywords: keywords indicating the differentes informations to log, they might be chosen by the user
        or specific for use special features of logger: ex: performance or model
         list_subsets: list of data subset with distinguished results: ex ["train", "eval", "test"] or [train, "test"]
        """
        self.root_log = root_log
        self.list_keywords = list_keywords
        self.list_subsets = list_subsets

        assert self.list_keywords is not None
        assert self.list_subsets is not None
        assert len(self.list_keywords) >= 1
        assert len(self.list_subsets) >= 1

        self.logger_dict = {}

        # create dict base
        for keyword in self.list_keywords:
            self.logger_dict[keyword] = {}
            for subset in self.list_subsets:
                self.logger_dict[keyword][subset] = []

        self.current_task = 0
        self.current_epoch = 0

        # add task and epochs in architecture
        self._update_dict_architecture(update_task=True)

    def add(self, value, keyword="performance", subset="train"):

        assert keyword in self.list_keywords, f"Keyword {keyword} is not declared in list_keywords {self.list_keywords}"
        assert subset in self.list_subsets, f"Field {subset} is not declared in list_keywords {self.list_subsets}"

        if keyword == "performance":
            predictions, targets, task_ids = value
            self._add_perf(predictions, targets, task_ids, subset)
        elif keyword == "model":
            self._add_model(model=value)
        else:
            self._add_value(value, keyword, subset)

    def _get_current_predictions(self, subset="train"):
        return self.logger_dict["performance"][subset][self.current_task][self.current_epoch]["predictions"]

    def _get_current_targets(self, subset="train"):
        return self.logger_dict["performance"][subset][self.current_task][self.current_epoch]["targets"]

    def _get_current_task_ids(self, subset="train"):
        return self.logger_dict["performance"][subset][self.current_task][self.current_epoch]["task_ids"]


    def _add_model(self, model):
        """
        we do not save model in logger we save it in memory
        """
        assert self.root_log is not None
        model2save = deepcopy(model).cpu().state_dict()
        filename = f"Model_epoch_{self.current_epoch}_Task_{self.current_task}.pth"
        filename = os.path.join(self.root_log, filename)
        torch.save(model2save, filename)

    def _add_value(self, _tensor, keyword, subset="train"):
        """
        we assume here that value is a tensor or a single value
        """

        _tensor = convert_numpy(_tensor)

        self.logger_dict[keyword][subset][self.current_task][self.current_epoch].append(
            _tensor)

    def _add_perf(self, predictions, targets, task_ids=None, subset="train"):
        """
        Special function for performance, so performance can be logged in one line
        """
        predictions = convert_numpy(predictions)
        targets = convert_numpy(targets)
        task_ids = convert_numpy(task_ids)

        if not isinstance(predictions, np.ndarray):
            raise TypeError(f"Provide predictions as np.ndarray, not {type(predictions).__name__}.")
        if not isinstance(targets, np.ndarray):
            raise TypeError(f"Provide targets as np.ndarray, not {type(targets).__name__}.")

        assert predictions.size == targets.size, f"{predictions.size} vs {targets.size}"

        predictions = np.concatenate([self._get_current_predictions(subset), predictions])
        self.logger_dict["performance"][subset][self.current_task][self.current_epoch]["predictions"] = predictions

        targets = np.concatenate([self._get_current_targets(subset), targets])
        self.logger_dict["performance"][subset][self.current_task][self.current_epoch]["targets"] = targets

        if (task_ids is not None) and self._get_current_task_ids(subset).size > 0:
            task_ids = np.concatenate([self._get_current_task_ids(subset), task_ids])
            self.logger_dict["performance"][subset][self.current_task][self.current_epoch]["task_ids"] = task_ids

    def _update_dict_architecture(self, update_task=False):
        for keyword in self.list_keywords:
            for subset in self.list_subsets:
                if update_task:
                    self.logger_dict[keyword][subset].append([])
                if keyword == "performance":
                    self.logger_dict[keyword][subset][self.current_task].append({})
                    self.logger_dict[keyword][subset][self.current_task][self.current_epoch][
                        "predictions"] = np.zeros(0)
                    self.logger_dict[keyword][subset][self.current_task][self.current_epoch]["targets"] = np.zeros(0)
                    self.logger_dict[keyword][subset][self.current_task][self.current_epoch]["task_ids"] = np.zeros(0)
                else:
                    self.logger_dict[keyword][subset][self.current_task].append([])

                assert len(self.logger_dict[keyword][subset])-1 == self.current_task
                assert len(self.logger_dict[keyword][subset][self.current_task])-1 == self.current_epoch

    def end_epoch(self):
        self.current_epoch += 1
        self._update_dict_architecture(update_task=False)

    def end_task(self):
        if self.root_log is not None:
            self._save_dic()
        self.current_task += 1
        self.current_epoch = 0
        self._update_dict_architecture(update_task=True)

    def _save_dic(self):
        import pickle as pkl
        filename = f"logger_dic_task_{self.current_task}.pkl"
        filename = os.path.join(self.root_log, filename)
        with open(filename, 'wb') as f:
            pkl.dump(self.logger_dict, f, pkl.HIGHEST_PROTOCOL)

        # after saving values we remove them from dictionnary to save space and memory
        for keyword in self.list_keywords:
            for subset in self.list_subsets:
                self.logger_dict[keyword][subset][self.current_task] = None
