from typing import Callable, List, Tuple, Union

import numpy as np
import torch
from clloader.datasets import BaseDataset
from clloader.viz import plot
from PIL import Image
from torchvision import transforms


class Dataset(torch.utils.data.Dataset):
    """A task dataset returned by the CLLoader.

    :param x: The data, either image-arrays or paths to images saved on disk.
    :param y: The targets, not one-hot encoded.
    :param trsf: The transformations to apply on the images.
    :param open_image: Whether to open image from disk, or index in-memory.
    """

    def __init__(
        self, x: np.ndarray, y: np.ndarray, trsf: transforms.Compose, open_image: bool = False
    ):
        self.x, self.y = x, y
        self.trsf = trsf
        self.open_image = open_image

    @property
    def nb_classes(self):
        """The number of classes contained in the current task."""
        return len(np.unique(self.y))

    def add_memory(self, x_memory: np.ndarray, y_memory: np.ndarray):
        """Add memory for rehearsal.

        :param x_memory: Sampled data chosen for rehearsal.
        :param y_memory: The associated targets of `x_memory`.
        """
        self.x = np.concatenate((self.x, x_memory))
        self.y = np.concatenate((self.y, y_memory))

    def plot(self, path=None, figsize=None, nb_per_class=5):
        """Plot samples of the current task, useful to check if everything is ok.

        :param path: If not None, save on disk at this path.
        :param figsize: The size of the figure.
        :param nb_per_class: Amount to sample per class.
        """
        plot(self, figsize=figsize, path=path, nb_per_class=nb_per_class)

    def __len__(self):
        """The amount of images in the current task."""
        return self.x.shape[0]

    def get_image(self, index):
        """Returns a Pillow image corresponding to the given `index`.

        :param index: Index to query the image.
        :return: A Pillow image.
        """
        x = self.x[index]
        if self.open_image:
            img = Image.open(x).convert("RGB")
        else:
            img = Image.fromarray(x.astype("uint8"))
        return img

    def __getitem__(self, index):
        """Method used by PyTorch's DataLoaders to query a sample and its target."""
        img = self.get_image(index)
        y = self.y[index]
        img = self.trsf(img)
        return img, y


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
    """

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

    def _setup(self) -> None:
        self.train_data, self.test_data = self.cl_dataset.init()
        unique_classes = np.unique(self.train_data[1])

        self.class_order = self.cl_dataset.class_order or list(range(len(unique_classes)))

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

    def __next__(self) -> Tuple[Dataset, Dataset]:
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
        elif split == "val":
            pass  # TODO
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
