import warnings
from typing import Callable, List, Tuple, Union

import numpy as np
from PIL import Image

import torch
from clloader.datasets import BaseDataset
from torchvision import transforms


class Dataset(torch.utils.data.Dataset):

    def __init__(
        self, x: np.ndarray, y: np.ndarray, trsf: transforms.Compose, open_image: bool = False
    ):
        self.x, self.y = x, y
        self.trsf = trsf
        self.open_image = open_image

    def add_memory(self, x_memory: np.ndarray, y_memory: np.ndarray):
        self.x = np.concatenate((self.x, x_memory))
        self.y = np.concatenate((self.y, y_memory))

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        x, y = self.x[idx], self.y[idx]

        if self.open_image:
            img = Image.open(x).convert("RGB")
        else:
            img = Image.fromarray(x.astype("uint8"))

        img = self.trsf(img)

        return img, y


class CLLoader:

    def __init__(
        self,
        cl_dataset: BaseDataset,
        increment: Union[List[int], int],
        initial_increment: int = 0,
        train_transformations: List[Callable] = None,
        common_transformations: List[Callable] = None,
        evaluate_on="seen"
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

        self._setup()

        self.increments = self.define_increments(increment, initial_increment)

    def define_increments(self, increment: Union[List[int], int],
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

    def _setup(self) -> None:
        self.train_data, self.test_data = self.cl_dataset.init()
        unique_classes = np.unique(self.train_data[1])

        self.class_order = self.cl_dataset.class_order or list(range(len(unique_classes)))

    @property
    def nb_classes(self) -> int:
        return len(np.unique(self.train_data[1]))

    @property
    def nb_tasks(self) -> int:
        return len(self)

    def __len__(self) -> int:
        """Returns the number of tasks.

        :return: Number of tasks.
        """
        return len(self.increments)

    def __iter__(self):
        self._counter = 0
        return self

    def __next__(self) -> Tuple[Dataset, Dataset]:
        if self._counter >= len(self):
            raise StopIteration
        task = self[self._counter]
        self._counter += 1
        return task

    def __getitem__(self, task_index):
        max_class = sum(self.increments[:task_index + 1])
        min_class = sum(self.increments[:task_index])  # 0 when task_index == 0.

        train = self._select_data(min_class, max_class)
        train_dataset = Dataset(*train, self.train_trsf, open_image=not self.cl_dataset.in_memory)

        # TODO: validation
        if self.evaluate_on == "seen":
            test = self._select_data(0, max_class, split="test")
        elif self.evaluate_on == "current":
            test = self._select_data(min_class, max_class, split="test")
        else:  # all
            test = self._select_data(0, self.nb_classes, split="test")

        test_dataset = Dataset(*test, self.test_trsf, open_image=not self.cl_dataset.in_memory)

        return train_dataset, test_dataset

    def _select_data(self, min_class, max_class, split="train"):
        if split == "train":
            x, y = self.train_data
        elif split == "val":
            pass  # TODO
        else:
            x, y = self.test_data

        indexes = np.where(np.logical_and(y >= min_class, y < max_class))[0]
        return x[indexes], y[indexes]
