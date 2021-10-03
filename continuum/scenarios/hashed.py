import os
from multiprocessing import Pool, cpu_count
from typing import Callable, List, Union, Optional

import imagehash
import numpy as np
from PIL import Image
from sklearn.metrics import pairwise_distances
from sklearn.cluster import KMeans, MeanShift
from sklearn.decomposition import PCA
from scipy.spatial.distance import hamming

from continuum.datasets import InMemoryDataset
from continuum.datasets import _ContinuumDataset
from continuum.scenarios import ContinualScenario
from continuum.tasks import TaskType


def sort_hash(list_hash):
    size = len(list_hash)

    # multithread hash evaluation without changing list order
    with Pool(min(8, cpu_count())) as p:
        list_hash_norm = p.map(np.linalg.norm, list_hash)

    assert len(list_hash_norm)==size

    return sorted(range(size), key=lambda k: list_hash_norm[k])


def get_array_list(list_bin_str_hash):
    list_array = []
    for str_hash in list_bin_str_hash:
        array = np.array([int(hex_str, 16) for hex_str in str_hash])
        list_array.append(array)

    return list_array


class HashedScenario(ContinualScenario):
    """Continual Loader, generating datasets for the consecutive tasks.

    Scenario: the scenario is entirely defined by the task label vector in the cl_dataset

    :param cl_dataset: A continual dataset.
    :param hash_name: Name of the hash function that will be used to create scenario
    :param nb_tasks: Number of tasks if we do not want to be set automatically
    :param transformations: A list of transformations applied to all tasks. If
                            it's a list of list, then the transformation will be
                            different per task.
    :param filename_hash_indexes: file to save scenarios indexes and reload them after to win some time and computation
    :param split_task: Define if the task split will be automatic by clusterization ("auto") of hashes
    or manually balanced ("balanced"). NB: if split_task == "balanced" nb_tasks can not be None.
    """

    def __init__(
            self,
            cl_dataset: _ContinuumDataset,
            hash_name,
            nb_tasks=None,
            transformations: Union[List[Callable], List[List[Callable]]] = None,
            filename_hash_indexes: Optional[str] = None,
            split_task="balanced"
    ) -> None:
        self.hash_name = hash_name
        self.split_task = split_task
        self._nb_tasks = nb_tasks

        if self.hash_name not in ["AverageHash", "Phash", "PhashSimple", "DhashH", "DhashV", "Whash", "ColorHash"
                                  ]: # , "CropResistantHash"
            AssertionError(f"{self.hash_name} is not a hash_name available.")
        if self.split_task not in ["balanced", "auto"]:
            AssertionError(f"{self.split_task} is not a data_split parameter available.")
        if split_task == "balanced" and nb_tasks is None:
            AssertionError(f"self.data_split is {self.split_task} the nb_tasks should be set.")

        self.data_type = cl_dataset.data_type
        self.filename_hash_indexes = filename_hash_indexes

        # "CropResistantHash" does not work yet
        # if self.hash_name == "CropResistantHash":
        #     # auto (kmeans) does not work with hask format of CropResistantHash
        #     self.split_task = "balanced"

        x, y, t = self.generate_task_ids(cl_dataset)
        cl_dataset = InMemoryDataset(x, y, t, data_type=self.data_type)
        super().__init__(cl_dataset=cl_dataset, transformations=transformations)

    def process_for_hash(self, x):
        """"preprocess data for hashing functions"""
        if self.data_type == TaskType.IMAGE_ARRAY:
            im = Image.fromarray(x.astype("uint8"))
        elif self.data_type == TaskType.IMAGE_PATH:
            im = Image.open(x).convert("RGB")
        else:
            raise NotImplementedError(f"data_type -- {self.data_type}"
                                      f" -- Not implemented or not Compatible")

        return im

    def hash_func(self, x):
        ''''Hash one image and return hash'''

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
            hash_value = imagehash.whash(x,
                                         hash_size=8,
                                         image_scale=None,
                                         mode='haar',
                                         remove_max_haar_ll=True)
        elif self.hash_name == "ColorHash":
            hash_value = imagehash.colorhash(x, binbits=3)
        elif self.hash_name == "CropResistantHash": # does not work yet
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

    def get_task_ids(self, x):
        '''Return the task id vectors corresponding to the parameters settings '''

        if self.split_task == "balanced":
            # In this case: create task ids with a fix set of tasks with a balanced amount of data
            assert self._nb_tasks is not None
            nb_examples = len(x)
            task_ids = np.ones(nb_examples) * (self._nb_tasks - 1)
            example_per_tasks = nb_examples // self._nb_tasks
            perfect_balance_task_ids = np.arange(self._nb_tasks).repeat(example_per_tasks)
            task_ids[:len(perfect_balance_task_ids)] = perfect_balance_task_ids

            # examples from len(perfect_balance_task_ids) to len(task_ids) are put into last tasks
        elif self._nb_tasks is not None:
            # In this case: create task ids with a fix set of tasks with a amount of data automatically set

            # we use KMeans from scikit learn to make hash coherent tasks with a fixed number of task

            # reduce data size for clustering
            pca = PCA(n_components=2)
            reduc_data = pca.fit_transform(x)

            # we use kmeans from scikit learn to create coherent clusters
            task_ids = KMeans(n_clusters=self._nb_tasks).fit_predict(reduc_data)
        else:
            # In this case: create task ids with an automatically set  number of tasks
            # with a amount of data automatically set.

            # we use MeanShift from scikit learn to automatically set the number of task
            # and make hash coherent tasks

            # reduce data size for clustering
            pca = PCA(n_components=2)
            reduc_data = pca.fit_transform(x)

            bandwidth = 5
            task_ids = None
            while (bandwidth > 0.5):
                task_ids = MeanShift(bandwidth=bandwidth, bin_seeding=True).fit_predict(reduc_data)
                if len(np.unique(task_ids)) > 3:
                    # we would like more that 3 tasks if possible
                    break
                else:
                    # reduce the bandwidth if there is not enough tasks
                    bandwidth = bandwidth * 0.75

            self._nb_tasks = len(np.unique(task_ids))
            if not self._nb_tasks > 1:
                AssertionError("The number of task is expected to be more than one.")

        assert len(np.unique(task_ids)) > 1, print(np.unique(task_ids))

        return task_ids

    def get_list_hash_ids(self, x):
        '''Compute hash for all data points in x and return a list of all hash in the same order '''

        # multithread hash evaluation without changing list order
        with Pool(min(8, cpu_count())) as p:
            list_hash = p.map(self.hash_func, list(x))
        return list_hash

    def generate_task_ids(self, cl_dataset):
        ''''This function handle the generation of task id, either by reloading self.filename_hash_indexes
        or by regenerating a task ids vector with self.get_task_ids(...)'''
        x, y, _ = cl_dataset.get_data()

        if self.filename_hash_indexes is not None and os.path.exists(self.filename_hash_indexes):
            print(f"Loading previously saved sorted indexes ({self.filename_hash_indexes}).")
            tuple_indexes_hash = np.load(self.filename_hash_indexes, allow_pickle=True)
            task_ids = tuple_indexes_hash[0].astype(int)
            sort_indexes = tuple_indexes_hash[1].astype(int)
            vectorized_list_hash = tuple_indexes_hash[2]

            assert len(sort_indexes) == len(vectorized_list_hash), print(
                f"sort_indexes {len(sort_indexes)} - list_hash {len(vectorized_list_hash)}")

            x = x[sort_indexes]
            y = y[sort_indexes]
        else:
            list_hash = self.get_list_hash_ids(x)

            vectorized_list_hash = get_array_list(list_hash)

            # arbitrary sorting based on vectorized hash norm
            sort_indexes = sort_hash(vectorized_list_hash)

            x = x[sort_indexes]
            y = y[sort_indexes]
            ordered_hash = np.array(vectorized_list_hash)[sort_indexes]
            task_ids = self.get_task_ids(ordered_hash)

            # save eventually sort_indexes for later use and gain of time
            if self.filename_hash_indexes is not None:
                np.save(self.filename_hash_indexes, [task_ids, sort_indexes, vectorized_list_hash], allow_pickle=True)

        if not len(task_ids) == len(y):
            AssertionError(f"task_ids {len(task_ids)} - y {len(y)} should be equal")

        return x, y, task_ids

    # nothing to do in the setup function
    def _setup(self, nb_tasks: int) -> int:
        return nb_tasks
