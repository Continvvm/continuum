import os
from typing import Tuple, Union, Optional

import numpy as np

from continuum.datasets.base import _AudioDataset
from continuum.download import download, untar
from continuum.tasks import TaskType


class FluentSpeech(_AudioDataset):
    URL = "http://fluent.ai:2052/jf8398hf30f0381738rucj3828chfdnchs.tar.gz"

    def __init__(self, data_path, train: Union[bool, str] = True, download: bool = True):
        if not isinstance(train, bool) and train not in ("train", "valid", "test"):
            raise ValueError(f"`train` arg ({train}) must be a bool or train/valid/test.")
        if isinstance(train, bool):
            if train:
                train = "train"
            else:
                train = "test"

        data_path = os.path.expanduser(data_path)
        super().__init__(data_path, train, download)

    def _download(self):
        if not os.path.exists(os.path.join(self.data_path, "fluent_speech_commands_dataset")):
            tgz_path = os.path.join(self.data_path, "jf8398hf30f0381738rucj3828chfdnchs.tar.gz")

            if not os.path.exists(tgz_path):
                print("Downloading tgz archive...", end=" ")
                download(
                    self.URL,
                    self.data_path
                )
                print("Done!")

            print("Extracting archive...", end=" ")
            untar(tgz_path)
            print("Done!")

    def get_data(self) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        audioid, transcriptions, intents, subintent = [], [], [], [[] for i in range(3)]
        base_path = os.path.join(self.data_path, "fluent_speech_commands_dataset")

        with open(os.path.join(base_path, "data", f"{self.train}_data.csv"), encoding="utf-8") as fcsv:
            lines = fcsv.readlines()

        for line in lines[1:]:
            items = line[:-1].split(",")
            audioid.append(os.path.join(base_path, items[1]))
            if (len(items)) == 7:
                transcriptions.append(items[3])
            else:
                transcriptions.append((" ").join(items[3:5]))
            intents.append(tuple(items[-3:]))
            for i in range(3):
                subintent[i].append(intents[-1][i])

        subintent_sets = [sorted(list(set(subintent[i]))) for i in range(3)]
        subintent_labels = []
        for i in range(3):
            subintent_labels.append([subintent_sets[i].index(t) for t in subintent[i]])

        concat_labels = [str([subintent_labels[i][j] for i in range(3)]) for j in range(len(subintent_labels[0]))]
        unique_labels = sorted(list((set([concat_labels[i] for i in range(len(concat_labels))]))))

        y = [unique_labels.index(l) for l in concat_labels]

        return np.array(audioid), np.array(y), None
