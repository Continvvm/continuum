from typing import List, Tuple, Union

from PIL import Image
from skimage.transform import resize
import numpy as np

from continuum.datasets import CIFAR10, SVHN, FashionMNIST, MNIST, DTD, RainbowMNIST
from continuum.datasets import _ContinuumDataset


class CTRL(_ContinuumDataset):
    """CTRL generic class.

    Reference:
        * Efficient Continual Learning with Modular Networks and Task-Driven Priors
          Tom Veniat, Ludovic Denoyer, Marc'Aurelio Ranzato
          ICLR 2021

    To get a feeling about this class, please look at the actual implementation of
    the various instances of CTRL datasets (e.g. CTRLminus, CTRLplus, etc.)

    :param datasets: The list of continual datasets to use.
    :param target_size: The common image size for all.
    :param split: Which split among train/val/test.
    :param proportions: How much images (in absolute value or %) to take per dataset.
    :param class_counter: The initial class counter per dataset, helps when seeing
                          twice the same dataset we want them to have the same or
                          different labels.
    :param class_subsets: A subset of classes to sample from the i-th dataset.
    :param seed: A random seed for reproducibility.
    """
    def __init__(
        self,
        datasets: List[_ContinuumDataset],
        target_size: Tuple[int, int] = (32, 32),
        split: str = "train",
        proportions: Union[None, List[int], List[float]] = None,
        class_counter: Union[None, List[int]] = None,
        class_subsets: Union[None, List[int]] = None,
        seed: int = 1
    ):
        class_counter = class_counter or [0 for _ in range(len(datasets))]
        class_subsets = class_subsets or [None for _ in range(len(datasets))]
        proportions = proportions or [None for _ in range(len(datasets))]
        if len(proportions) != len(datasets):
            raise ValueError(
                f"There must have as much proportions {len(proportions)} as datasets ({len(datasets)})."
            )
        if len(class_counter) != len(datasets):
            raise ValueError(
                f"There must have as much class counters {len(class_counter)} as datasets ({len(datasets)})."
            )
        if split not in ("train", "val", "test"):
            raise ValueError(f"Split must be train, val, or test; not {split}.")

        self.datasets = datasets
        self.target_size = target_size
        self.split = split
        self.proportions = proportions
        self.class_counter = class_counter
        self.class_subsets = class_subsets
        self.seed = seed

        super().__init__()

    def get_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        all_x, all_y, all_t = [], [], []

        task_id = 0
        for i, dataset in enumerate(self.datasets):
            x, y, _ = dataset.get_data()

            if self.class_subsets[i] is not None:
                indexes = np.where(np.isin(y, self.class_subsets[i]))[0]
                x, y = x[indexes], y[indexes]
            if self.split in ("train", "val") and self.proportions[i]:
                indexes = self.balanced_sampling(y, self.proportions[i], self.seed, self.split)
                x, y = x[indexes], y[indexes]
            if x.dtype == "S255":  # String paths
                x = self.open_and_resize(x, self.target_size)
            if len(x.shape) == 3:  # Grayscale images
                x = x[..., None]
            if x.shape[-1] == 1:  # Grayscale images
                x = np.repeat(x, 3, axis=-1)
            if x.shape[1:3] != self.target_size:  # Not the correct widh or height
                x = self.resize(x, self.target_size)

            all_x.append(x)
            all_y.append(y + self.class_counter[i])
            all_t.append(np.ones(len(y)) * task_id)
            task_id += 1

        return np.concatenate(all_x), np.concatenate(all_y), np.concatenate(all_t)

    def open_and_resize(self, paths: np.ndarray, size: Tuple[int, int]):
        x = np.zeros((len(paths), *size, 3), dtype=np.uint8)
        for i, path in enumerate(paths):
            img = Image.open(path).convert('RGB').resize(size)
            x[i] = np.array(img).astype(np.uint8)
        return x

    def resize(self, arrays: np.ndarray, size: Tuple[int, int]):
        new_arrays = np.zeros((len(arrays), *size, 3), dtype=np.uint8)
        for i, arr in enumerate(arrays):
            new_arrays[i] = resize(arr, size, preserve_range=True).astype(np.uint8)
        return new_arrays

    def balanced_sampling(self, y: np.ndarray, amount: Union[float, int], seed: int, split: str = "train"):
        """Samples a certain amount of data equally per class."""
        if isinstance(amount, float):
            amount = int(len(y) * amount)
        unique_classes = np.unique(y)
        if len(unique_classes) > amount:
            raise ValueError(
                f"Not enough amount ({amount}) for the number of classes ({len(unique_classes)})."
            )
        # There can be a few images lost, but that's not very important.
        amount_per_class = int(amount / len(unique_classes))

        rng = np.random.RandomState(seed=seed)

        indexes = []
        for c in unique_classes:
            class_indexes = np.where(y == c)[0]
            rng.shuffle(class_indexes)

            if split == "train":
                indexes.append(class_indexes[:amount_per_class])
            else:  # val
                indexes.append(class_indexes[-amount_per_class:])

        return np.concatenate(indexes)



class CTRLminus(CTRL):  # S^-
    def __init__(self, data_path: str = "", split: str = "train", download: bool = True, seed: int = 1):
        if split not in ("train", "val", "test"):
            raise ValueError(f"Split must be train, val, or test; not {split}.")
        train = split in ("train", "val")

        datasets = [
            CIFAR10(data_path=data_path, train=train, download=download),
            MNIST(data_path=data_path, train=train, download=download),
            DTD(data_path=data_path, train=train, download=download),
            FashionMNIST(data_path=data_path, train=train, download=download),
            SVHN(data_path=data_path, train=train, download=download),
            CIFAR10(data_path=data_path, train=train, download=download)
        ]

        if split == "train":
            proportions = [4000, 400, 400, 400, 400, 400]
        elif split == "val":
            proportions = [2000, 200, 200, 200, 200, 200]
        else:
            proportions = None

        super().__init__(
            datasets=datasets,
            proportions=proportions,
            class_counter=[0, 10, 20, 67, 77, 0],
            seed=seed,
            split=split
        )


class CTRLplus(CTRL):  # S^+
    def __init__(self, data_path: str = "", split: str = "train", download: bool = True, seed: int = 1):
        if split not in ("train", "val", "test"):
            raise ValueError(f"Split must be train, val, or test; not {split}.")
        train = split in ("train", "val")

        datasets = [
            CIFAR10(data_path=data_path, train=train, download=download),
            MNIST(data_path=data_path, train=train, download=download),
            DTD(data_path=data_path, train=train, download=download),
            FashionMNIST(data_path=data_path, train=train, download=download),
            SVHN(data_path=data_path, train=train, download=download),
            CIFAR10(data_path=data_path, train=train, download=download)
        ]

        if split == "train":
            proportions = [400, 400, 400, 400, 400, 4000]
        elif split == "val":
            proportions = [200, 200, 200, 200, 200, 2000]
        else:
            proportions = None

        super().__init__(
            datasets=datasets,
            proportions=proportions,
            class_counter=[0, 10, 20, 67, 77, 0],
            seed=seed,
            split=split
        )


class CTRLin(CTRL):  # S^{in}
    def __init__(self, data_path: str = "", split: str = "train", download: bool = True, seed: int = 1):
        if split not in ("train", "val", "test"):
            raise ValueError(f"Split must be train, val, or test; not {split}.")
        train = split in ("train", "val")

        color1, color2 = np.random.RandomState(seed=seed).choice(["red", "blue", "green"], 2)
        datasets = [
            RainbowMNIST(data_path=data_path, train=train, download=download, color=color1),
            CIFAR10(data_path=data_path, train=train, download=download),
            DTD(data_path=data_path, train=train, download=download),
            FashionMNIST(data_path=data_path, train=train, download=download),
            SVHN(data_path=data_path, train=train, download=download),
            RainbowMNIST(data_path=data_path, train=train, download=download, color=color2)
        ]

        if split == "train":
            proportions = [4000, 400, 400, 400, 400, 50]
        elif split == "val":
            proportions = [2000, 200, 200, 200, 200, 30]
        else:
            proportions = None

        super().__init__(
            datasets=datasets,
            proportions=proportions,
            class_counter=[0, 10, 20, 67, 77, 0],
            seed=seed,
            split=split
        )


class CTRLout(CTRL):  # S^{out}
    def __init__(self, data_path: str = "", split: str = "train", download: bool = True, seed: int = 1):
        if split not in ("train", "val", "test"):
            raise ValueError(f"Split must be train, val, or test; not {split}.")
        train = split in ("train", "val")

        datasets = [
            CIFAR10(data_path=data_path, train=train, download=download),
            MNIST(data_path=data_path, train=train, download=download),
            DTD(data_path=data_path, train=train, download=download),
            FashionMNIST(data_path=data_path, train=train, download=download),
            SVHN(data_path=data_path, train=train, download=download),
            CIFAR10(data_path=data_path, train=train, download=download)
        ]

        if split == "train":
            proportions = [4000, 400, 400, 400, 400, 400]
        elif split == "val":
            proportions = [2000, 200, 200, 200, 200, 200]
        else:
            proportions = None

        super().__init__(
            datasets=datasets,
            proportions=proportions,
            class_counter=[0, 10, 20, 67, 77, 87],
            seed=seed,
            split=split
        )


class CTRLplastic(CTRL):  # S^{pl}
    def __init__(self, data_path: str = "", split: str = "train", download: bool = True, seed: int = 1):
        if split not in ("train", "val", "test"):
            raise ValueError(f"Split must be train, val, or test; not {split}.")
        train = split in ("train", "val")

        datasets = [
            MNIST(data_path=data_path, train=train, download=download),
            DTD(data_path=data_path, train=train, download=download),
            FashionMNIST(data_path=data_path, train=train, download=download),
            SVHN(data_path=data_path, train=train, download=download),
            CIFAR10(data_path=data_path, train=train, download=download)
        ]

        if split == "train":
            proportions = [400, 400, 400, 400, 4000]
        elif split == "val":
            proportions = [200, 200, 200, 200, 2000]
        else:
            proportions = None

        super().__init__(
            datasets=datasets,
            proportions=proportions,
            class_counter=[0, 10, 57, 67, 77],
            seed=seed,
            split=split
        )


# TODO: add CTRLlong
