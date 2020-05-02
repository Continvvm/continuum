import json
import os

import numpy as np

from clloader.datasets.base import _ContinuumDataset


class MultiNLI(_ContinuumDataset):

    def __init__(self, data_path: str = "", download: bool = False) -> None:
        super().__init__(data_path, download)

        if self.download:
            self._download()

    def _download(self):
        if os.path.exists(self.data_path):
            print("MultiNLI already downloaded.")
        else:
            raise IOError(
                "You must download the dataset at this address: "
                "https://www.nyu.edu/projects/bowman/multinli/multinli_1.0.zip"
                " and unzip it."
            )

    @property
    def data_type(self) -> str:
        return "text"

    @property
    def transformations(self):
        return []

    def original_targets(self):
        return ["contradiction", "entailment", "neutral"]

    def init(self, train: bool):
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

        if train:
            json_path = os.path.join(self.data_path, "multinli_1.0_train.jsonl")
        else:
            json_path = os.path.join(self.data_path, "multinli_1.0_dev_mismatched.jsonl")

        with open(json_path) as f:
            for line in f:
                line = json.loads(line)

                if line["gold_label"] not in available_targets:
                    continue
                texts.append((line["sentence1"], line["sentence2"]))
                targets.append(available_targets.index(line["gold_label"]))

                if train:
                    genres.append(available_genres.index(line["genre"]))
                else:
                    genres.append(0)

        texts = np.array(texts)
        targets = np.array(targets)
        genres = np.array(genres)

        return texts, targets, genres
