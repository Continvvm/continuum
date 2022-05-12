import os
import collections
import itertools
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
        base_path = os.path.join(self.data_path, "fluent_speech_commands_dataset")

        self.class_ids = collections.defaultdict(itertools.count().__next__)
        self.actions = collections.defaultdict(itertools.count().__next__)
        self.objs = collections.defaultdict(itertools.count().__next__)
        self.locations = collections.defaultdict(itertools.count().__next__)
        self.speakerids = collections.defaultdict(itertools.count().__next__)
        self.transcriptions = []

        x, y, t = [], [], []

        with open(os.path.join(base_path, "data", f"{self.train}_data.csv")) as f:
            lines = f.readlines()[1:]

        for line in lines:
            items = line[:-1].split(',')

            action, obj, location = items[-3:]

            x.append(os.path.join(base_path, items[1]))
            y.append([
                self.class_ids[action+obj+location],
                self.actions[action],
                self.objs[obj],
                self.locations[location]
            ])
            t.append(self.speakerids[items[2]])

            self.transcriptions.append(items[3])

        return np.array(x), np.array(y), np.array(t)
