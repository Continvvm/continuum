import os
import glob
from typing import Optional, Tuple, Dict, Callable

import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
import torch

from continuum import download
from continuum.datasets.base import _SegmentationDataset


id_to_trainid = {  # Creating a dummy value for background at 0
    0: 255, # unlabeled
    1: 255, # ego
    2: 255, # rectification border
    3: 255, # out of roi
    4: 255, # static
    5: 255, # dynamic
    6: 255, # ground
    7: 1,   # road
    8: 2,   # sidewalk
    9: 255, # parking
    10: 255, # rail track
    11: 3,  # building
    12: 4,  # wall
    13: 5,  # fence
    14: 255, # guard rail
    15: 255, # bridge
    16: 255, # tunnel
    17: 6,  # pole
    18: 255, # polegroup
    19: 7,  # traffic light
    20: 8,  # traffic sign
    21: 9,  # vegetation
    22: 10,  # terrain
    23: 11, # sky
    24: 12, # person
    25: 13, # rider
    26: 14, # car
    27: 15, # truck
    28: 16, # bus
    29: 255, # caravan
    30: 255, # trailer
    31: 17, # train
    32: 18, # motorcycle
    33: 19, # bicycle
    -1: 255 # license plate
}


cities = [
    # train
    "aachen", "bochum", "bremen", "cologne", "darmstadt", "dusseldorf",  "erfurt",
    "hamburg", "hanover", "jena", "krefeld", "monchengladbach", "strasbourg",
    "stuttgart", "tubingen", "ulm", "weimar", "zurich",
    # val
    "frankfurt", "lindau", "munster"
]


class CityScapes(_SegmentationDataset):
    """CityScapes Semantic Segmentation Dataset.

    :param data_path: Path where the data is present or will be downloaded.
    :param download: Whether to download.
    """

    def __init__(
        self,
        data_path: str = "",
        train: bool = True,
        download: bool = True,
        test_split: Optional[float] = None,
        random_seed: int = 1,
        panoptic: bool = False
    ) -> None:
        super().__init__(data_path=data_path, train=train, download=download)

        self.test_split = test_split
        self.random_seed = random_seed
        self._panoptic = panoptic

    @property
    def nb_classes(self):
        return 19

    @property
    def class_map(self) -> Dict[int, int]:
        return id_to_trainid

    @property
    def panoptic(self) -> bool:
        return self._panoptic

    def prepare(self, seg_map: torch.Tensor) -> torch.Tensor:
        if self.panoptic:
            seg_map = self._create_panoptic(seg_map)
            seg_map[0].apply_(lambda x: self.class_map[x])
        else:
            seg_map.apply_(lambda x: self.class_map[x])
        return seg_map

    def _create_panoptic(self, seg_map: torch.Tensor) -> torch.Tensor:
        """Disentangle the semantic (class info) from instance info.

        :param seg_map: A single-channel segmentation map, for a single image,
                        with instance ids encoded as class id * 1000 + instance id.
        :return: A two-channel segmentation map, first channel for class ids,
                    and second channel for instance ids.
        """
        new_seg_map = torch.ones((2, seg_map.shape[0], seg_map.shape[1]), dtype=seg_map.dtype) * 255

        things = (seg_map >= 1000).bool()
        new_seg_map[0][things] = seg_map[things]
        new_seg_map[0][~things] = seg_map[~things] // 1000

        new_seg_map[1][~things] = seg_map[~things] % 1000

        return new_seg_map

    def _download(self):
        # Downloading ground-truth annotations
        if not os.path.exists(os.path.join(self.data_path, "gtFine")):
            path = os.path.join(self.data_path, "gtFine_trainvaltest.zip")
            if not os.path.exists(path):
                raise IOError(
                    "Please download yourself the file gtFine_trainvaltest.zip"
                    " at the URL: https://www.cityscapes-dataset.com/downloads/."
                    " Note that you may need to create an account."
                )
            download.unzip(path)

        # Downloading images
        if not os.path.exists(os.path.join(self.data_path, "leftImg8bit")):
            path = os.path.join(self.data_path, "leftImg8bit_trainvaltest.zip")
            if not os.path.exists(path):
                raise IOError(
                    "Please download yourself the file leftImg8bit_trainvaltest.zip"
                    " at the URL: https://www.cityscapes-dataset.com/downloads/."
                    " Note that you may need to create an account."
                )

    def get_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.train and self.test_split is None:
            x, y, t = _parse_dataset(self.data_path, "train", self.panoptic)
        elif not self.train and self.test_split is None:
            x, y, t = _parse_dataset(self.data_path, "val", self.panoptic)
        else:
            x_tr, y_tr, t_tr = _parse_dataset(self.data_path, "train", self.panoptic)
            x_va, y_va, t_va = _parse_dataset(self.data_path, "val",  self.panoptic)

            x = np.array(x_tr + x_va)
            y = np.array(y_tr + y_va)
            t = np.array(t_tr + t_va)

            sss = StratifiedShuffleSplit(
                n_splits=1,  test_size=self.test_split, random_state=self.random_seed)

            indexes_tr, indexes_va = next(sss.split(None, t))

            if self.train:
                x, y, t = x[indexes_tr], y[indexes_tr], t[indexes_tr]
            else:
                x, y, t = x[indexes_va], y[indexes_va], t[indexes_va]

        return np.array(x), np.array(y), np.array(t)


def _parse_dataset(
    data_path: str,
    split: str = "train",
    panoptic: bool = False
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    x, y, t = [], [], []

    label_type = "instance" if panoptic else "label"
    label_end = f"*gtFine_{label_type}Ids.png"

    x_base_path = os.path.join(data_path, "leftImg8bit", split)
    y_base_path = os.path.join(data_path, "gtFine", split)
    for city_name in os.listdir(x_base_path):
        if city_name not in cities:
            continue

        city_id = cities.index(city_name)
        images = glob.glob(os.path.join(x_base_path, city_name, "*.png"))
        maps = glob.glob(os.path.join(y_base_path, city_name, label_end))

        if len(images) != len(maps):
            raise IOError(
                f"For city {city_name}, there're {len(images)} images for {len(maps)} maps."
            )

        x.extend(images)
        y.extend(maps)
        t.extend([city_id for _ in range(len(images))])

    return x, y, t
