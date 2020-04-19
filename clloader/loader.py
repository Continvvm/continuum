from typing import Callable, List, Tuple, Union

import numpy as np
from clloader.datasets import BaseDataset, ContinuumDataset
from torch.utils.data import Dataset as TorchDataset
from torchvision import transforms

class CLLoader:
    """Continual Loader, generating datasets for the consecutive tasks.

    :param cl_dataset: A continual dataset.
    :param increment: Either number of classes per task, or a list specifying for
                      every task the amount of new classes.
    :param initial_increment: A different task size applied only for the first task.
                              Desactivated if `increment` is a list.
    :param train_transformations: A list of data augmentation applied to the train set.
    :param common_transformations: A list of transformations applied to both the
                                   the train set and the test set. i.e. normalization,
                                   resizing, etc.
    :param evaluate_on: How to evaluate on val/test, either on all `seen` classes,
                        on the `current` classes, or on `all` classes.
    :param class_order: An optional custom class order, used for NC.
    """

    def __init__(
        self,
        cl_dataset: BaseDataset,
        increment: Union[List[int], int],
        initial_increment: int = 0,
        train_transformations: List[Callable] = None,
        common_transformations: List[Callable] = None,
        evaluate_on="seen",
        class_order=None
    ) -> None:
        self.cl_dataset = cl_dataset

        if train_transformations is None:
            train_transformations = []
        if common_transformations is None:
            common_transformations = self.cl_dataset.transformations
        self.train_trsf = transforms.Compose(train_transformations + common_transformations)
        self.test_trsf = transforms.Compose(common_transformations)

        if evaluate_on not in ("seen", "current", "all"):
            raise NotImplementedError(f"Evaluate mode {evaluate_on} is not supported.")
        self.evaluate_on = evaluate_on

        self._setup(class_order)

        self.increments = self._define_increments(increment, initial_increment)

    def _define_increments(self, increment: Union[List[int], int],
                           initial_increment: int) -> List[int]:
        if isinstance(increment, list):
            return increment
        increments = []
        if initial_increment:
            increments.append(initial_increment)

        nb_tasks = (self.nb_classes - initial_increment) / increment
        if not nb_tasks.is_integer():
            raise Exception(
                "The tasks won't have an equal number of classes"
                f" with {len(self.class_order)} and increment {increment}"
            )
        increments.extend([increment for _ in range(int(nb_tasks))])

        return increments

    def _setup(self, class_order: Union[None, List[int]] = None) -> None:
        (train_x, train_y), (test_x, test_y) = self.cl_dataset.init()
        unique_classes = np.unique(train_y)

        self.class_order = class_order or self.cl_dataset.class_order or list(
            range(len(unique_classes))
        )
        if len(np.unique(self.class_order)) != len(self.class_order):
            raise ValueError(f"Invalid class order, duplicates found: {self.class_order}.")

        mapper = np.vectorize(lambda x: self.class_order.index(x))
        train_y = mapper(train_y)
        test_y = mapper(test_y)

        self.train_data = (train_x, train_y)
        self.test_data = (test_x, test_y)
        self.class_order = np.array(self.class_order)

    def get_original_targets(self, targets: np.ndarray) -> np.ndarray:
        """Returns the original targets not changed by the custom class order.

        :param targets: An array of targets, as provided by the task datasets.
        :return: An array of targets, with their original values.
        """
        return self.class_order[targets]

    @property
    def nb_classes(self) -> int:
        """Total number of classes in the whole continual setting."""
        return len(np.unique(self.train_data[1]))

    @property
    def nb_tasks(self) -> int:
        """Number of tasks in the whole continual setting."""
        return len(self)

    def __len__(self) -> int:
        """Returns the number of tasks.

        :return: Number of tasks.
        """
        return len(self.increments)

    def __iter__(self):
        """Used for iterating through all tasks with the CLLoader in a for loop."""
        self._counter = 0
        return self

    def __next__(self) -> Tuple[ContinuumDataset, ContinuumDataset]:
        """An iteration/task in the for loop."""
        if self._counter >= len(self):
            raise StopIteration
        task = self[self._counter]
        self._counter += 1
        return task

    def __getitem__(self, task_index):
        """Returns a task by its unique index.

        :param task_index: The unique index of a task, between 0 and len(loader) - 1.
        :return: A train and test PyTorch's Datasets.
        """
        max_class = sum(self.increments[:task_index + 1])
        min_class = sum(self.increments[:task_index])  # 0 when task_index == 0.

        train = self._select_data(min_class, max_class)
        train_dataset = ContinuumDataset(*train, self.train_trsf, open_image=not self.cl_dataset.in_memory)

        # TODO: validation
        if self.evaluate_on == "seen":
            test = self._select_data(0, max_class, split="test")
        elif self.evaluate_on == "current":
            test = self._select_data(min_class, max_class, split="test")
        else:  # all
            test = self._select_data(0, self.nb_classes, split="test")

        test_dataset = ContinuumDataset(*test, self.test_trsf, open_image=not self.cl_dataset.in_memory)

        return train_dataset, test_dataset

    def _select_data(self, min_class, max_class, split="train"):
        """Selects a subset of the whole data for a given task.

        :param min_class: The minimum class id.
        :param max_class: The maximum class id.
        :param split: Either sample from the `train` set, the `val` set, or the
                      `test` set.
        :return: A tuple of numpy array, the first item being the data and the
                 second the associated targets.
        """
        if split == "train":
            x, y = self.train_data
        else:
            x, y = self.test_data

        indexes = np.where(np.logical_and(y >= min_class, y < max_class))[0]
        selected_x = x[indexes]
        selected_y = y[indexes]

        if self.cl_dataset.need_class_remapping:
            # A remapping of the class ids is done to handle some special cases
            # like PermutedMNIST or RotatedMNIST.
            selected_y = self.cl_dataset.class_remapping(selected_y)

        return selected_x, selected_y


def split_train_val(dataset: TorchDataset,
                    val_split: float = 0.1) -> Tuple[TorchDataset, TorchDataset]:
    """Split train dataset into two datasets, one for training and one for validation.

    :param dataset: A torch dataset, with .x and .y attributes.
    :param val_split: Percentage to allocate for validation, between [0, 1[.
    :return: A tuple a dataset, respectively for train and validation.
    """
    random_state = np.random.RandomState(seed=1)

    indexes = np.arange(len(dataset.x))
    random_state.shuffle(indexes)

    train_indexes = indexes[int(val_split * len(indexes)):]
    val_indexes = indexes[:int(val_split * len(indexes))]

    x, y = dataset.x, dataset.y
    train_dataset = ContinuumDataset(x[train_indexes], y[train_indexes], dataset.trsf, dataset.open_image)
    val_dataset = ContinuumDataset(x[val_indexes], y[val_indexes], dataset.trsf, dataset.open_image)

    return train_dataset, val_dataset
