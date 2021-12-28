from typing import List, Union, Tuple

import numpy as np
import datasets
from datasets.arrow_dataset import Dataset as HFDataset

from continuum.scenarios import _BaseScenario


class HuggingFaceContinual(_BaseScenario):
    def __init__(
        self,
        hf_dataset: Union[HFDataset, str, Tuple],
        split_field: str,
        increment: int = 1,
        train: bool = True
    ):
        self.split_field = split_field
        self.split = "train" if train else "test"
        self.increment = increment

        if isinstance(hf_dataset, str):
            self.hf_dataset = datasets.load_dataset(hf_dataset, split=self.split)
        elif isinstance(hf_dataset, tuple):
            self.hf_dataset = datasets.load_dataset(*hf_dataset, split=self.split)
        else:
            self.hf_dataset = hf_dataset

        self._classes = np.unique(self.hf_dataset[split_field])

    def _setup(self):
        pass

    @property
    def train(self) -> bool:
        """Returns whether we are in training or testing mode."""
        return self.split == "train"

    @property
    def nb_samples(self) -> int:
        """Total number of samples in the whole continual setting."""
        return len(self.hf_dataset)

    @property
    def nb_classes(self) -> int:
        return len(self.classes)

    @property
    def classes(self) -> List:
        return self._classes

    def __len__(self):
        return self.nb_classes // self.increment

    def __getitem__(self, index):
        classes = set(self.classes[index * self.increment:(index + 1) * self.increment])
        return self.hf_dataset.filter(lambda x: x[self.split_field] in classes)
