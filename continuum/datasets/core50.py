import os
from typing import Iterable, Set, Tuple, Union

import numpy as np

from continuum import download
from continuum.datasets.base import _ContinuumDataset


class Core50(_ContinuumDataset):
    """Continuum version of the Core50 dataset.

    References:
        * Core50: a new Dataset and Benchmark for Continuous Object Recognition
          Lomonaco & Maltoni.
          CoRL 2017

    :param data_path: The folder path containing the data.
    :param train_image_ids: The image ids belonging to the train set. Either the
                            csv file containing them auto-downloaded, or a list
                            of string.
    :param download: An option useless in this case.
    """

    data_url = "http://bias.csr.unibo.it/maltoni/download/core50/core50_128x128.zip"
    train_ids_url = "https://vlomonaco.github.io/core50/data/core50_train.csv"

    def __init__(
        self,
        data_path: str,
        train_image_ids: Union[str, Iterable[str], None] = None,
        download: bool = True
    ):
        self.train_image_ids = train_image_ids
        super().__init__(data_path, download)

        if isinstance(self.train_image_ids, str):
            self.train_image_ids = self._read_csv(self.train_image_ids)
        elif isinstance(self.train_image_ids, list):
            self.train_image_ids = set(self.train_image_ids)

    @property
    def data_type(self):
        return "image_path"

    def _download(self):
        if os.path.exists(os.path.join(self.data_path, "core50_128x128")):
            print("Dataset already extracted.")
        else:
            path = download.download(self.data_url, self.data_path)
            download.unzip(path)
            print("Dataset extracted.")

        split_path = os.path.join(self.data_path, "core50_train.csv")
        if self.train_image_ids is None and os.path.exists(split_path):
            self.train_image_ids = split_path
            print("Train/split already downloaded.")
        elif self.train_image_ids is None:
            print("Downloading train/test split.")
            self.train_image_ids = download.download(self.train_ids_url, self.data_path)

    def _read_csv(self, csv_file: str) -> Set[str]:
        """Read the csv file containing the ids of training samples."""
        train_images_ids = set()

        with open(csv_file, "r") as f:
            print(csv_file)
            for line in f:
                image_id = line.split(",")[0].split(".")[0]
                train_images_ids.add(image_id)

        return train_images_ids

    def init(self, train: bool) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate the CORe50 data.

        CORe50, in one of its many iterations, is made of 50 objects, each present
        in 10 different domains (in-door, street, garden, etc.).

        In class incremental (NC) setting, those domains won't matter.

        In instance incremental (NI) setting, the domains come one after the other,
        but all classes are present since the first task. Seven domains are allocated
        for the train set, while 3 domains are allocated for the test set.

        In the case of the test set, all domains have the "dummy" label of 0. The
        authors designed this dataset with a fixed test dataset in mind.
        """
        x, y, t = [], [], []

        domain_counter = 0
        for domain_id in range(10):
            # We walk through the 10 available domains.
            domain_folder = os.path.join(self.data_path, "core50_128x128", f"s{domain_id + 1}")

            has_images = False
            for object_id in range(50):
                # We walk through the 50 available object categories.
                object_folder = os.path.join(domain_folder, f"o{object_id + 1}")

                for path in os.listdir(object_folder):
                    image_id = path.split(".")[0]

                    if (
                        (train and image_id not in self.train_image_ids) or  # type: ignore
                        (not train and image_id in self.train_image_ids)  # type: ignore
                    ):
                        continue

                    x.append(os.path.join(object_folder, path))
                    y.append(object_id)
                    if train:  # We add a new domain id for the train set.
                        t.append(domain_counter)
                    else:  # Test set is fixed, therefore we artificially give a unique domain.
                        t.append(0)

                    has_images = True
            if has_images:
                domain_counter += 1

        x = np.array(x)
        y = np.array(y)
        t = np.array(t)

        return x, y, t


class Core50v2_79(_ContinuumDataset):
    data_url = "http://bias.csr.unibo.it/maltoni/download/core50/core50_128x128.zip"
    splits_url = "https://vlomonaco.github.io/core50/data/batches_filelists_NICv2.zip"
    nb_tasks = 79

    def __init__(self, data_path: str, download: bool = True, run_id: int = 0):
        if run_id > 9 or run_id < 0:
            raise ValueError(
                "CORe50 v2 only provides split for 10 runs (ids 0 to 9),"
                f" invalid run_id={run_id}."
            )
        self.run_id = run_id

        super().__init__(data_path, download)

    def _download(self):
        if os.path.exists(os.path.join(self.data_path, "core50_128x128")):
            print("Dataset already extracted.")
        else:
            path = download.download(self.data_url, self.data_path)
            download.unzip(path)
            print("Dataset extracted.")

        if os.path.exists(os.path.join(self.data_path, "batches_filelists_NICv2.zip")):
            print("Split info already downloaded.")
        else:
            path = download.download(self.splits_url, self.data_path)
            download.unzip(path)
            print("Split info extracted.")

    def init(self, train: bool) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if train:
            return self._train_init()
        return self._test_init()

    def _test_init(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        text_file = os.path.join(
            self.data_path, f"NIC_v2_{self.nb_tasks}", f"run{self.run_id}", "test_filelist.txt"
        )
        paths, targets = self._read_txt(text_file)

        return paths, targets, np.zeros(len(targets))

    def _train_init(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        template = os.path.join(
            self.data_path, f"NIC_v2_{self.nb_tasks}", f"run{self.run_id}",
            "train_batch_{}_filelist.txt"
        )

        paths, targets, tasks = [], [], []
        for task_id in range(self.nb_tasks):
            p, t = self._read_txt(template.format(str(task_id).rjust(2, "0")))

            paths.append(p)
            targets.append(t)
            tasks.append(task_id * np.ones(len(t)))

        paths = np.concatenate(paths)
        targets = np.concatenate(targets)
        tasks = np.concatenate(tasks)

        return paths, targets, tasks

    def _read_txt(self, path: str) -> Tuple[np.ndarray, np.ndarray]:
        """Read Core50 v2 split info that are stored in a txt file.

        The format, for each line, is "path<space>class_id".

        :param path: The path to the text file.
        :return: An array of image paths and an array of targets.
        """
        image_root_path = os.path.join(self.data_path, "core50_128x128")

        paths, targets = [], []
        with open(path, "r") as f:
            for line in f:
                p, t = line.strip().split(" ")
                paths.append(os.path.join(image_root_path, p))
                targets.append(int(t))

        return np.array(paths), np.array(targets)


class Core50v2_196(Core50v2_79):
    nb_tasks = 196


class Core50v2_391(Core50v2_79):
    nb_tasks = 391
