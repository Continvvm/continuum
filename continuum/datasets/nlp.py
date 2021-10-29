import json
import os
from typing import Dict, List, Tuple

import numpy as np

from continuum import download
from continuum.datasets.base import _ContinuumDataset
from continuum.tasks import TaskType


class MultiNLI(_ContinuumDataset):
    """Continuum version of the MultiNLI dataset.

    References:
        * A Broad-Coverage Challenge Corpus for Sentence Understanding through Inference
          Williams, Nangia, and Bowman
          ACL 2018
        * Progressive Memory Banks for Incremental Domain Adaptation
          Asghar & Mou
          ICLR 2020

    The dataset is based on the NLI task.
    For each example, two sentences are given. The goal is to determine whether
    this pair of sentences has:
    - Opposite meaning (contradiction)
    - Similar meaning (entailment)
    - no relation to each other (neutral)

    :param data_path: The folder extracted from the official zip file.
    :param download: An option useless in this case.
    """

    data_url = "https://cims.nyu.edu/~sbowman/multinli/multinli_1.0.zip"

    def __init__(self, data_path: str = "", train: bool = True, download: bool = True) -> None:
        super().__init__(data_path=data_path, train=train, download=download)

    def _download(self):
        if os.path.exists(os.path.join(self.data_path, "multinli_1.0")):
            print("Dataset already extracted.")
        else:
            path = download.download(self.data_url, self.data_path)
            download.unzip(path)
            print("Dataset extracted.")

    @property
    def nb_classes(self):
        return 20

    @property
    def data_type(self) -> TaskType:
        return TaskType.TEXT

    @property
    def transformations(self):
        return []

    def original_targets(self) -> List[str]:
        return ["contradiction", "entailment", "neutral"]

    def get_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate the MultiNLI data.

        The dataset has several domains, but always the same targets
        ("contradiction", "entailment", "neutral"). 5 domains are allocated for
        the train set ("fiction", "government", "slate", "telephone", "travel"),
        and 5 to the test set ("facetoface", "letters", "nineeleven", "oup",
        "verbatim").

        While the train is given different task id for each domain, the test set
        always has a dummy 0 domain id, as it is supposed to be fixed.
        """
        texts, targets, genres = [], [], []

        available_targets = ["contradiction", "entailment", "neutral"]
        available_genres = [
            "fiction",
            "government",
            "slate",
            "telephone",
            "travel",  # /train
            "facetoface",
            "letters",
            "nineeleven",
            "oup",
            "verbatim"  # /test
        ]

        if self.train:
            json_path = os.path.join(self.data_path, "multinli_1.0", "multinli_1.0_train.jsonl")
        else:
            json_path = os.path.join(
                self.data_path, "multinli_1.0", "multinli_1.0_dev_mismatched.jsonl"
            )

        with open(json_path) as f:
            for line in f:
                line_parsed: Dict[str, str] = json.loads(line)

                if line_parsed["gold_label"] not in available_targets:
                    continue  # A few cases exist w/o targets.

                texts.append((line_parsed["sentence1"], line_parsed["sentence2"]))
                targets.append(available_targets.index(line_parsed["gold_label"]))

                if self.train:  # We add a new domain id for the train set.
                    genres.append(available_genres.index(line_parsed["genre"]))
                else:  # Test set is fixed, therefore we artificially give a unique domain.
                    genres.append(0)

        texts = np.array(texts)
        targets = np.array(targets)
        genres = np.array(genres)

        return texts, targets, genres
