import os
import json
from typing import Tuple, Optional

from torchvision import datasets as torchdata
import numpy as np

from continuum.datasets import _ContinuumDataset
from continuum.download import download, unzip


class Stream51(_ContinuumDataset):
    """Stream51 dataset.

    Reference:
        * Stream-51: Streaming Classification and Novelty Detection From Videos
          Roady, Ryne and Hayes, Tyler L. and Vaidya, Hitesh and Kanan, Christopher
          CVPR Workshops 2020

    Official implementation here: https://github.com/tyler-hayes/Stream-51

    :param data_path: The folder path containing the data.
    :param train: Train or Test mode.
    :param download: Auto-download of the dataset if it doesn't exist.
    :param crop: Uses the bounding boxes to crop objects intead of taking whole images.
    :param ratio: A ratio factor to take a bit more pixels than just the bounding box.
    """

    url = "http://klab.cis.rit.edu/files/Stream-51.zip"

    def __init__(
        self,
        data_path: str = "",
        train: bool = True,
        download: bool = True,
        crop: bool = True,
        ratio : float = 1.1
    ):
        super().__init__(data_path=data_path, train=train, download=download)

        self.crop = crop
        self.ratio = ratio
        self._bounding_boxes = None

    def _download(self):
        path = os.path.join(self.data_path, "Stream-51")
        if not os.path.exists(f"{path}.zip"):
            download(self.url, self.data_path)
        if not os.path.exists(path):
            unzip(f"{path}.zip")

        print("Stream51 downloaded.")

    def get_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        x, y, t, b = self._parse_json()

        if self.crop:
            self._set_bounding_boxes(b)

        return x, y, t

    def _set_bounding_boxes(self, bounding_boxes: np.ndarray):
        """Format the bounding boxes in order to enlarge them through ratio param.

        See official code about it, there:
        https://github.com/tyler-hayes/Stream-51/blob/master/experiments/StreamDataset.py#L139

        :param bounding_boxes: A list of bounding boxes as provided by the dataset.
        """
        formatted_bounding_boxes = []

        for bbox in bounding_boxes:
            cw = bbox[0] - bbox[1]
            ch = bbox[2] - bbox[3]
            center = [int(bbox[1] + cw / 2), int(bbox[3] + ch / 2)]
            formatted_bounding_boxes.append([
                int(center[0] - (cw * self.ratio / 2)), # x1
                int(center[1] - (ch * self.ratio / 2)), # y1
                int(center[0] + (cw * self.ratio / 2)), # x2
                int(center[1] + (ch * self.ratio / 2)), # y2
            ])

        self._bounding_boxes = np.array(formatted_bounding_boxes)

    @property
    def bounding_boxes(self):
        return self._bounding_boxes

    @property
    def data_type(self) -> str:
        return "image_path"

    def _parse_json(self):
        if self.train:
            path = os.path.join(self.data_path, "Stream-51", "Stream-51_meta_train.json")
        else:
            path = os.path.join(self.data_path, "Stream-51", "Stream-51_meta_test.json")

        # X: path, Y: class id, T: task id (aka video id), B: bounding box
        x, y, t, b = [], [], [], []
        with open(path) as f:
            data = json.load(f)

        for line in data:
            x.append(os.path.join(self.data_path, "Stream-51", line[-1]))
            y.append(line[0])
            if self.train:
                t.append(line[2])
            else:
                t.append(0)
            b.append(line[-2])

        return np.array(x), np.array(y), np.array(t), np.array(b)
