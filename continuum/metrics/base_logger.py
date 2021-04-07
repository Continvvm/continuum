import abc
import os
import torch
import numpy as np


def convert_numpy(tensor):
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.cpu().numpy()
    return tensor


class _BaseLogger(abc.ABC):
    def __init__(self, root_log=None, list_keywords=["performance"], list_subsets=["train", "test"]):
        """
        root_log: folder where logged informations will be saved
        list_keywords: keywords indicating the differentes informations to log, they might be chosen by the user
        or specific for use special features of logger: ex: performance or model_size
         list_subsets: list of data subset with distinguished results: ex ["train", "val", "test"] or [train, "test"]
        """
        self.root_log = root_log
        self.list_keywords = list_keywords
        self.list_subsets = list_subsets

        assert self.list_keywords is not None, f" self.list_keywords should contains at list one keyword"
        assert self.list_subsets is not None, f" self.list_subsets should contains at list one subset"
        assert len(self.list_keywords) >= 1, f" self.list_keywords should contains at list one keyword"
        assert len(self.list_subsets) >= 1, f" self.list_subsets should contains at list one subset"
        assert len(self.list_subsets) == len(set(self.list_subsets)), f" There are duplicate in the subset list"

        self.logger_dict = {}

        # create dict base
        for subset in self.list_subsets:
            self.logger_dict[subset] = {}
            for keyword in self.list_keywords:
                self.logger_dict[subset][keyword] = []

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
        else:
            self._add_value(value, keyword, subset)

    def _get_current_predictions(self, subset="train"):
        return self.logger_dict[subset]["performance"][self.current_task][self.current_epoch]["predictions"]

    def _get_current_targets(self, subset="train"):
        return self.logger_dict[subset]["performance"][self.current_task][self.current_epoch]["targets"]

    def _get_current_task_ids(self, subset="train"):
        return self.logger_dict[subset]["performance"][self.current_task][self.current_epoch]["task_ids"]

    def _add_value(self, tensor, keyword, subset="train"):
        """Add a tensor in the list of the current epoch (tensor can also be a single value) """

        tensor = convert_numpy(tensor)

        self.logger_dict[subset][keyword][self.current_task][self.current_epoch].append(
            tensor)

    def _add_perf(self, predictions, targets, task_ids=None, subset="train"):
        """Special function for performance, so performance can be logged in one line """
        predictions = convert_numpy(predictions)
        targets = convert_numpy(targets)
        task_ids = convert_numpy(task_ids)

        if not isinstance(predictions, np.ndarray):
            raise TypeError(f"Provide predictions as np.ndarray, not {type(predictions).__name__}.")
        if not isinstance(targets, np.ndarray):
            raise TypeError(f"Provide targets as np.ndarray, not {type(targets).__name__}.")

        assert predictions.size == targets.size, f"{predictions.size} vs {targets.size}"

        predictions = np.concatenate([self._get_current_predictions(subset), predictions])
        self.logger_dict[subset]["performance"][self.current_task][self.current_epoch]["predictions"] = predictions

        targets = np.concatenate([self._get_current_targets(subset), targets])
        self.logger_dict[subset]["performance"][self.current_task][self.current_epoch]["targets"] = targets

        if task_ids is not None:
            task_ids = np.concatenate([self._get_current_task_ids(subset), task_ids])
            self.logger_dict[subset]["performance"][self.current_task][self.current_epoch]["task_ids"] = task_ids

    def _update_dict_architecture(self, update_task=False):
        for keyword in self.list_keywords:
            for subset in self.list_subsets:
                if update_task:
                    self.logger_dict[subset][keyword].append([])
                if keyword == "performance":
                    self.logger_dict[subset][keyword][self.current_task].append({})
                    self.logger_dict[subset][keyword][self.current_task][self.current_epoch][
                        "predictions"] = np.zeros(0)
                    self.logger_dict[subset][keyword][self.current_task][self.current_epoch]["targets"] = np.zeros(0)
                    self.logger_dict[subset][keyword][self.current_task][self.current_epoch]["task_ids"] = np.zeros(0)
                else:
                    self.logger_dict[subset][keyword][self.current_task].append([])

                assert len(self.logger_dict[subset][keyword]) - 1 == self.current_task, \
                    f"the current task index is {self.current_task} while there are" \
                    f" {len(self.logger_dict[subset][keyword]) - 1} past tasks, there is a mismatch"
                assert len(self.logger_dict[subset][keyword][self.current_task]) - 1 == self.current_epoch, \
                    f"the current epoch index is {self.current_epoch} while there are" \
                    f" {len(self.logger_dict[subset][keyword][self.current_task]) - 1} past epoch, there is a mismatch"

    def print_state(self, keyword, subset):

        print(self.logger_dict)

        print(f"**********************")
        print(f"{keyword} on {subset}")
        print(f"**********************")
        for task in range(self.current_task):
            print(f"**********************")
            print(f"Task: {task}")
            for epoch in range(self.current_epoch + 1):
                print(f"Epoch: {task}")
                print(self.logger_dict[subset][keyword][task][epoch])

    def end_epoch(self):
        self.current_epoch += 1
        self._update_dict_architecture(update_task=False)

    def end_task(self, clean=False):
        if self.root_log is not None and clean:
            self._save()
        self.current_task += 1
        self.current_epoch = 0
        self._update_dict_architecture(update_task=True)

    def _save(self):
        import pickle as pkl
        filename = f"logger_dic_task_{self.current_task}.pkl"
        filename = os.path.join(self.root_log, filename)
        with open(filename, 'wb') as f:
            pkl.dump(self.logger_dict, f, pkl.HIGHEST_PROTOCOL)

        # after saving values we remove them from dictionary to save space and memory
        for subset in self.list_subsets:
            for keyword in self.list_keywords:
                self.logger_dict[subset][keyword][self.current_task] = None
