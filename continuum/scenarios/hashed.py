import warnings
from copy import copy
from typing import Callable, List, Union

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
            data_shape=None,
            transformations: Union[List[Callable], List[List[Callable]]] = None,
    ) -> None:
        self.hash_name = hash_name
        self.data_shape = data_shape
        self.data_type = cl_dataset.data_type
        x, y, t = self.generate_task_ids(cl_dataset, self.hash_name, nb_tasks)
        cl_dataset = InMemoryDataset(x, y, t, data_type=self.data_type)
        super().__init__(cl_dataset=cl_dataset, transformations=transformations)

    def process_for_hash(self, x):
        if self.data_type == "image_array":
            # x = np.array(x)
            # x = x.reshape(self.data_shape)
            # x = x - x.min()
            # x = x / x.max()
            # x = np.uint8(x * 255)
            # im = Image.fromarray(x)
            im = Image.fromarray(x.astype("uint8"))
        elif self.data_type == "image_path":
            im = Image.open(x).convert("RGB")
        else:
            raise NotImplementedError(f"data_type -- {self.data_type} -- Not implemented or not Compatible")

        return im

    def hash_func(self, x, hash_name):

        x = self.process_for_hash(x)

        if hash_name == "AverageHash":
            hash_value = imagehash.average_hash(x, hash_size=8, mean=np.mean)
        elif hash_name == "Phash":
            hash_value = imagehash.phash(x, hash_size=8, highfreq_factor=4)
        elif hash_name == "PhashSimple":
            hash_value = imagehash.phash_simple(x, hash_size=8, highfreq_factor=4)
        elif hash_name == "DhashH":
            hash_value = imagehash.dhash(x)
        elif hash_name == "DhashV":
            hash_value = imagehash.dhash_vertical(x)
        elif hash_name == "Whash":
            hash_value = imagehash.whash(x, hash_size=8, image_scale=None, mode='haar', remove_max_haar_ll=True)
        elif hash_name == "ColorHash":
            hash_value = imagehash.colorhash(x, binbits=3)
        elif hash_name == "CropResistantHash":
            hash_value = imagehash.crop_resistant_hash(x,
                                                       hash_func=None,
                                                       limit_segments=None,
                                                       segment_threshold=128,
                                                       min_segment_size=500,
                                                       segmentation_image_size=300
                                                       )
        else:
            raise NotImplementedError(f"Hash Name -- {hash_name} -- Unknown")

        return hash_value

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

    def generate_task_ids(self, cl_dataset, hash_name, nb_tasks):
        x, y, t = cl_dataset.get_data()

        list_hash = []
        for i in range(len(y)):
            list_hash.append(str(self.hash_func(x[i], hash_name)))

        sort_indexes = self.sort_hash(list_hash)

        x = x[sort_indexes]
        y = y[sort_indexes]
        task_ids = self.get_task_ids(len(sort_indexes), nb_tasks)

        assert len(task_ids) == len(y)
        return x, y, task_ids

    def save_list_ids(self):
        # TODO: it might avoid to recompute the hash everytime
        pass

    # nothing to do in the setup function
    def _setup(self, nb_tasks: int) -> int:
        return nb_tasks
