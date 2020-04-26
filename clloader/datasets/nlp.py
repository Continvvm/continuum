import collections
import json

import numpy as np

from clloader.datasets.base import _ContinuumDataset


class MultiNLI(_ContinuumDataset):

    def __init__(self, data_path: str = "", download: bool = True) -> None:
        super().__init__(data_path, download)

        if self.download:
            self._download()

    def _download(self):
        raise IOError(
            "You must download the dataset at this address: "
            "https://www.nyu.edu/projects/bowman/multinli/multinli_1.0.zip"
            " and unzip it."
        )

    def original_targets(self):
        return ["contradiction", "entailment", "neutral"]

    def init(self, train: bool):
        texts, targets, genres = [], [], []

        available_targets = ["contradiction", "entailment", "neutral"]
        available_genres = ["fiction", "government", "slate", "telephone", "travel"]

        with open(self.data_path) as f:
            for line in f:
                line = json.loads(line)

                texts.append((line["sentence1"], line["sentence2"]))
                targets.append(available_targets.index(line["gold_label"]))
                genres.append(available_genres.index(line["genre"]))

        texts = np.array(texts)
        targets = np.array(targets)
        genres = np.array(genres)

        return texts, targets, genres
