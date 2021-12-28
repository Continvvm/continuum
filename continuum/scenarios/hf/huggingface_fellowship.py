from typing import List, Union, Tuple

import datasets
from datasets.arrow_dataset import Dataset as HFDataset

from continuum.scenarios import _BaseScenario


class HuggingFaceFellowship(_BaseScenario):
    """A scenario for a collection of HuggingFace (HF) dataset.

    It simply wraps multiple datasets and returns them one by one.
    To have a full list of the available datasets (only HuggingFace!), see
    there: https://huggingface.co/datasets

    :param hf_datasets: A list of HF dataset instances or a list of HF dataset string id.
    :param lazy: Load datasets on-the-fly when needed.
    :param train: Train split vs test split.
    """
    def __init__(
        self,
        hf_datasets: Union[List[HFDataset], List[str], List[Tuple]],
        lazy: bool = False,
        train: bool = True
    ):
        self.hf_datasets = hf_datasets
        self.lazy = lazy
        self.split = "train" if train else "test"

    def _setup(self):
        pass

    @property
    def train(self) -> bool:
        """Returns whether we are in training or testing mode."""
        return self.split == "train"

    @property
    def nb_samples(self) -> int:
        """Total number of samples in the whole continual setting."""
        if self.lazy:
            raise Exception("Cannot tell the number of samples if datasets are lazyly loaded.")
        return sum(len(dataset) for dataset in self.hf_datasets)

    @property
    def nb_classes(self) -> int:
        raise NotImplementedError("Not available for this kind of scenario.")

    @property
    def classes(self) -> List:
        raise NotImplementedError("Not available for this kind of scenario.")

    def __len__(self):
        return len(self.hf_datasets)

    def __getitem__(self, index):
        if self.lazy:
            if isinstance(self.hf_datasets[index], tuple):
                return datasets.load_dataset(*self.hf_datasets[index], split=self.split)
            return datasets.load_dataset(self.hf_datasets[index], split=self.split)
        return self.hf_datasets[index]
