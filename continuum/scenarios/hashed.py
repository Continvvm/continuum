import warnings
from copy import copy
from typing import Callable, List, Union, Optional
import os
from multiprocessing import Pool, cpu_count

import numpy as np
from PIL import Image
import imagehash

from continuum.datasets import _ContinuumDataset
from continuum.datasets import InMemoryDataset
from continuum.scenarios import ContinualScenario


class HashedScenario(ContinualScenario):
    """Continual Loader, generating datasets for the consecutive tasks.

    Scenario: the scenario is entirely defined by the task label vector in the cl_dataset

    :param cl_dataset: A continual dataset.
    :param transformations: A list of transformations applied to all tasks. If
                            it's a list of list, then the transformation will be
                            different per task.
    """

    def __init__(
            self,
            cl_dataset: _ContinuumDataset,
            hash_name,
            nb_tasks=None,
            transformations: Union[List[Callable], List[List[Callable]]] = None,
            filename_hash_indexes: Optional[str] = None,
    ) -> None:
        self.hash_name = hash_name
        self.data_type = cl_dataset.data_type
        self.filename_hash_indexes = filename_hash_indexes
        x, y, t = self.generate_task_ids(cl_dataset, nb_tasks)
        cl_dataset = InMemoryDataset(x, y, t, data_type=self.data_type)
        super().__init__(cl_dataset=cl_dataset, transformations=transformations)

    def process_for_hash(self, x):
        if self.data_type == "image_array":
            im = Image.fromarray(x.astype("uint8"))
        elif self.data_type == "image_path":
            im = Image.open(x).convert("RGB")
        else:
            raise NotImplementedError(f"data_type -- {self.data_type} -- Not implemented or not Compatible")

        return im

    def hash_func(self, x):

        x = self.process_for_hash(x)

        if self.hash_name == "AverageHash":
            hash_value = imagehash.average_hash(x, hash_size=8, mean=np.mean)
        elif self.hash_name == "Phash":
            hash_value = imagehash.phash(x, hash_size=8, highfreq_factor=4)
        elif self.hash_name == "PhashSimple":
            hash_value = imagehash.phash_simple(x, hash_size=8, highfreq_factor=4)
        elif self.hash_name == "DhashH":
            hash_value = imagehash.dhash(x)
        elif self.hash_name == "DhashV":
            hash_value = imagehash.dhash_vertical(x)
        elif self.hash_name == "Whash":
            hash_value = imagehash.whash(x, hash_size=8, image_scale=None, mode='haar', remove_max_haar_ll=True)
        elif self.hash_name == "ColorHash":
            hash_value = imagehash.colorhash(x, binbits=3)
        elif self.hash_name == "CropResistantHash":
            hash_value = imagehash.crop_resistant_hash(x,
                                                       hash_func=None,
                                                       limit_segments=None,
                                                       segment_threshold=128,
                                                       min_segment_size=500,
                                                       segmentation_image_size=300
                                                       )
        else:
            raise NotImplementedError(f"Hash Name -- {self.hash_name} -- Unknown")

        return str(hash_value)

    def sort_hash(self, list_hash):
        sort_indexes = sorted(range(len(list_hash)), key=lambda k: list_hash[k])
        return sort_indexes

    def get_task_ids(self, nb_examples, nb_tasks):
        task_ids = np.ones(nb_examples) * (nb_tasks - 1)

        example_per_tasks = nb_examples // nb_tasks
        perfect_balance_task_ids = np.arange(nb_tasks).repeat(example_per_tasks)
        task_ids[:len(perfect_balance_task_ids)] = perfect_balance_task_ids

        # examples from len(perfect_balance_task_ids) to len(task_ids) are affected to last tasks

        return task_ids

    def get_list_hash_ids(self, x):
        # multithread hash evaluation without changing list order
        with Pool(min(8, cpu_count())) as p:
            list_hash = p.map(self.hash_func, list(x))
        return list_hash

    def generate_task_ids(self, cl_dataset, nb_tasks):
        x, y, t = cl_dataset.get_data()

        if self.filename_hash_indexes is not None and os.path.exists(self.filename_hash_indexes):
            print(f"Loading previously saved sorted indexes ({self.filename_hash_indexes}).")
            sort_indexes = np.load(self.filename_hash_indexes)
        else:

            list_hash = self.get_list_hash_ids(x)
            sort_indexes = self.sort_hash(list_hash)

            # save eventually sort_indexes for later use and gain of time
            if self.filename_hash_indexes is not None:
                np.save(self.filename_hash_indexes, sort_indexes)

        x = x[sort_indexes]
        y = y[sort_indexes]
        task_ids = self.get_task_ids(len(sort_indexes), nb_tasks)
        assert len(task_ids) == len(y)

        return x, y, task_ids

    # nothing to do in the setup function
    def _setup(self, nb_tasks: int) -> int:
        return nb_tasks
