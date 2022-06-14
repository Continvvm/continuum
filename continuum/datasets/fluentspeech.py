import os
import collections
import itertools
from typing import Tuple, Union, Optional

import numpy as np

from continuum.datasets.base import _AudioDataset
from continuum.download import download, untar


class FluentSpeech(_AudioDataset):
    """FluentSpeechCommand dataset.

    Made of short audio with different speakers asking something.

    https://fluent.ai/fluent-speech-commands-a-dataset-for-spoken-language-understanding-research/
    """
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
                self.objects[obj],
                self.locations[location]
            ])
            if self.train == "train":
                t.append(self.train_speaker_ids[items[2]])
            elif self.train == "valid":
                t.append(self.valid_speaker_ids[items[2]])
            else:
                t.append(self.test_speaker_ids[items[2]])

            self.transcriptions.append(items[3])

        return np.array(x), np.array(y), np.array(t)

    @property
    def actions(self):
        return {
            'change language': 0,
            'activate': 1,
            'deactivate': 2,
            'increase': 3,
            'decrease': 4,
            'bring': 5
        }

    @property
    def objects(self):
        return {
            'none': 0,
            'music': 1,
            'lights': 2,
            'volume': 3,
            'heat': 4,
            'lamp': 5,
            'newspaper': 6,
            'juice': 7,
            'socks': 8,
            'Chinese': 9,
            'Korean': 10,
            'English': 11,
            'German': 12,
            'shoes': 13
        }

    @property
    def locations(self):
        return {
            'none': 0,
            'kitchen': 1,
            'bedroom': 2,
            'washroom': 3
        }

    @property
    def class_ids(self):
        return {
             'change languagenonenone': 0,
             'activatemusicnone': 1,
             'activatelightsnone': 2,
             'deactivatelightsnone': 3,
             'increasevolumenone': 4,
             'decreasevolumenone': 5,
             'increaseheatnone': 6,
             'decreaseheatnone': 7,
             'deactivatemusicnone': 8,
             'activatelampnone': 9,
             'deactivatelampnone': 10,
             'activatelightskitchen': 11,
             'activatelightsbedroom': 12,
             'activatelightswashroom': 13,
             'deactivatelightskitchen': 14,
             'deactivatelightsbedroom': 15,
             'deactivatelightswashroom': 16,
             'increaseheatkitchen': 17,
             'increaseheatbedroom': 18,
             'increaseheatwashroom': 19,
             'decreaseheatkitchen': 20,
             'decreaseheatbedroom': 21,
             'decreaseheatwashroom': 22,
             'bringnewspapernone': 23,
             'bringjuicenone': 24,
             'bringsocksnone': 25,
             'change languageChinesenone': 26,
             'change languageKoreannone': 27,
             'change languageEnglishnone': 28,
             'change languageGermannone': 29,
             'bringshoesnone': 30
        }

    @property
    def train_speaker_ids(self):
        return {
             '2BqVo8kVB2Skwgyb': 0,
             '2ojo7YRL7Gck83Z3': 1,
             '35v28XaVEns4WXOv': 2,
             '4aGjX3AG5xcxeL7a': 3,
             '52XVOeXMXYuaElyw': 4,
             '5BEzPgPKe8taG9OB': 5,
             '5o9BvRGEGvhaeBwA': 6,
             '5pa4DVyvN2fXpepb': 7,
             '73bEEYMKLwtmVwV43': 8,
             '7NEaXjeLX3sg3yDB': 9,
             '8e5qRjN7dGuovkRY': 10,
             '9EWlVBQo9rtqRYdy': 11,
             '9Gmnwa5W9PIwaoKq': 12,
             '9mYN2zmq7aTw4Blo': 13,
             'anvKyBjB5OiP5dYZ': 14,
             'AvR9dePW88IynbaE': 15,
             'AY5e3mMgZkIyG3Ox': 16,
             'BvyakyrDmQfWEABb': 17,
             'd2waAp3pEjiWgrDEY': 18,
             'd3erpmyk9yFlVyrZ': 19,
             'DMApMRmGq5hjkyvX': 20,
             'DWmlpyg93YCXAXgE': 21,
             'EExgNZ9dvgTE3928': 22,
             'eL2w4ZBD7liA85wm': 23,
             'eLQ3mNg27GHLkDej': 24,
             'g2dnA9Wpvzi2WAmZ': 25,
             'G3QxQd7qGRuXAZda': 26,
             'gNYvkbx3gof2Y3V9': 27,
             'gvKeNY2D3Rs2jRdL': 28,
             'Gym5dABePPHA8mZK9': 29,
             'jgxq52DoPpsR9ZRx': 30,
             'KLa5k73rZvSlv82X': 31,
             'kNnmb7MdArswxLYw': 32,
             'KqDyvgWm4Wh8ZDM7': 33,
             'kxgXN97ALmHbaezp': 34,
             'ldrknAmwYPcWzp4N': 35,
             'LR5vdbQgp3tlMBzB': 36,
             'M4ybygBlWqImBn9oZ': 37,
             'mor8vDGkaOHzLLWBp': 38,
             'mzgVQ4Z5WvHqgNmY': 39,
             'n5XllaB4gZFwZXkBz': 40,
             'neaPN7GbBEUex8rV': 41,
             'nO2pPlZzv3IvOQoP2': 42,
             'NWAAAQQZDXC5b9Mk': 43,
             'ObdQbr9wyDfbmW4E': 44,
             'OepoQ9jWQztn5ZqL': 45,
             'oNOZxyvRe3Ikx3La': 46,
             'oRrwPDNPlAieQr8Q': 47,
             'oXjpaOq4wVUezb3x': 48,
             'ppzZqYxGkESMdA5Az': 49,
             'qNY4Qwveojc8jlm4': 50,
             'R3mexpM2YAtdPbL7': 51,
             'R3mXwwoaX9IoRVKe': 52,
             'RjDBre8jzzhdr4YL': 53,
             'ro5AaKwypZIqNEp2': 54,
             'roOVZm7kYzS5d4q3': 55,
             'Rq9EA8dEeZcEwada2': 56,
             'rwqzgZjbPaf5dmbL': 57,
             'W4XOzzNEbrtZz4dW': 58,
             'W7LeKXje7QhZlLKe': 59,
             'wa3mwLV3ldIqnGnV': 60,
             'WYmlNV2rDkSaALOE': 61,
             'X4vEl3glp9urv4GN': 62,
             'xEYa2wgAQof3wyEO': 63,
             'xPZw23VxroC3N34k': 64,
             'xRQE5VD7rRHVdyvM': 65,
             'xwpvGaaWl5c3G5N3': 66,
             'xwzgmmv5ZOiVaxXz': 67,
             'Xygv5loxdZtrywr9': 68,
             'YbmvamEWQ8faDPx2': 69,
             'ywE435j4gVizvw3R': 70,
             'Z7BXPnplxGUjZdmBZ': 71,
             'zaEBPeMY4NUbDnZy': 72,
             'Ze7YenyZvxiB4MYZ': 73,
             'ZebMRl5Z7dhrPKRD': 74,
             'zwKdl7Z2VRudGj2L': 75,
             'zZezMeg5XvcbRdg3': 76,
        }

    @property
    def valid_speaker_ids(self):
        return {
            '7NqqnAOPVVSKnxyv':0,
            '8B9N9jOOXGUordVG':1,
            '9MX3AgZzVgCw4W4j':2,
            'D4jGxZ7KamfVo4E2V':3,
            'DWNjK4kYDACjeEg3':4,
            'eBQAWmMg4gsLYLLa':5,
            'mj4BWeRbp7ildyB9d':6,
            'NgXwdx5KkZI5GRWa':7,
            'Pz327QrLaGuxW8Do':8,
            'vnljypgejkINbBAY':9

        }

    @property
    def test_speaker_ids(self):
        return {
            '4BrX8aDqK2cLZRYl': 0,
            '7B4XmNppyrCK977p': 1,
            'aokxBz9LxXHzZzay': 2,
            'k5bqyxx2lzIbrlg9': 3,
            'NgQEvO2x7Vh3xy2xz': 4,
            'oOK5kxoW7dskMbaK': 5,
            'ppymZZDb2Bf4NQnE': 6,
            'Q4vMvpXkXBsqryvZ': 7,
            'V4ejqNL4xbUKkYrV': 8,
            'V4ZbwLm9G5irobWn': 9
        }
