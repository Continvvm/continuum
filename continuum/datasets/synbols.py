# from tqdm import tqdm
from torch.utils.data import Dataset
import numpy as np
import json
import os
from torchvision import transforms as tt
from PIL import Image
import torch
import multiprocessing
import sys
from continuum.datasets.base import InMemoryDataset
from functools import partial
import copy
import requests
import h5py


class Synbols(InMemoryDataset):
    categorical_attributes = ['char', 'font', 'alphabet', 'is_bold', 'is_slant']
    continuous_attributes = ['rotation', 'translation.x', 'translation.y', 'scale']
    def __init__(
            self,
            data_path: str,
            task_type: str = "char",
            domain_incremental_task: str = None,
            domain_increments: int = None,
            train: bool = True,
            dataset_name: str = "default_n=100000_2020-Oct-19.h5py",
            download: bool = True):
        """Wraps Synbols in a Continuum dataset for compatibility with Sequoia  

        Args:
            data_path (str): Path where the dataset will be saved
            train (bool): Whether to use the train split
            task_type (str): Options are 'char', and 'font'
            domain_incremental_task (str): The options are listed as static attribuets of the "Synbols" class.
            domain_increments (int): Amount of domain increments.
            dataset_name (str): See https://github.com/ElementAI/synbols-resources/raw/master/datasets/generated/',
            download (bool): Whether to download the dataset
        """
        assert(domain_incremental_task is None and domain_increments is None or \
            domain_incremental_task is not None and domain_increments is not None)

        if download:  # done here in order to pass x and y to super
            full_path = get_data_path_or_download("default_n=100000_2020-Oct-19.h5py",
                                                  data_root=data_path)
        else:
            full_path = os.path.join(data_path, dataset_name)

        data = SynbolsHDF5(full_path,
                           task_type,
                           domain_incremental_task=domain_incremental_task,
                           mask='random',
                           trim_size=None,
                           raw_labels=False)
        data = SynbolsSplit(data, 'train' if train else 'val', domain_incremental_task, domain_increments)

        super().__init__(data.x, data.y, data.task_id, train=train, download=download)


def _read_json_key(args):
    string, key = args
    return json.loads(string)[key]

def process_task(task, fields):
    data = json.loads(task)
    ret = []
    for field in fields:
        if '.x' in field:
            ret.append(data[field[:-2]][0])
        elif '.y' in field:
            ret.append(data[field[:-2]][1])
        else:
            ret.append(data[field])
    return ret

class SynbolsHDF5:
    """HDF5 Backend Class"""

    def __init__(self, path, task, domain_incremental_task=None, ratios=[0.6, 0.2, 0.2], mask=None, trim_size=None, raw_labels=False, reference_mask=None):
        """Constructor: loads data and parses labels.

        Args:
            path (str): path where the data is stored (see full_path above)
            task (str): 'char', 'font', or the field of choice 
            domain_incremental_task (str): task used for domain-incremental learning
            ratios (list, optional): The train/val/test split ratios. Defaults to [0.6, 0.2, 0.2].
            mask (ndarray, optional): Mask with the data partition. Defaults to None.
            trim_size (int, optional): Trim the dataset to a smaller size, for debugging speed. Defaults to None.
            raw_labels (bool, optional): Whether to include all the attributes of the synbols for each batch. Defaults to False.
            reference_mask (ndarray, optional): If train and validation are done with two different datasets, the 
                                                reference mask specifies the partition of the training data. Defaults to None.

        Raises:
            ValueError: Error message
        """
        self.path = path
        self.task = task
        self.domain_incremental_task = domain_incremental_task
        self.ratios = ratios
        print("Loading hdf5...")
        with h5py.File(path, 'r') as data:
            self.x = data['x'][...]
            y = data['y'][...]
            print("Converting json strings to labels...")
            parse_fields = [self.task]
            if self.domain_incremental_task is not None:
                parse_fields.append(self.domain_incremental_task)
            with multiprocessing.Pool(min(8, multiprocessing.cpu_count())) as pool:
                self.y = pool.map(partial(process_task, fields=parse_fields), y)
            self.y = list(map(np.array, zip(*self.y)))
            if self.domain_incremental_task is not None:
                self.y, self.domain_y = self.y
            else:
                self.y = self.y[0]
                self.domain_y = None
            print("Done converting.")

            self.mask = data["split"][mask][...]

            self.raw_labels = None

            self.trim_size = trim_size
            self.reference_mask = None
            print("Done reading hdf5.")

    def parse_mask(self, mask, ratios):
        return mask.astype(bool)

class SynbolsSplit(Dataset):
    def __init__(self, dataset, split, domain_incremental_task, domain_increments, transform=None):
        """Given a Backend (dataset), it splits the data in train, val, and test.


        Args:
            dataset (object): backend to load, it should contain the following attributes:
                - x, y, mask, ratios, path, task, mask
            split (str): train, val, or test
            domain_incremental_task (str): attribute used for domain-incremental learning
            domain_increments (int): number of domain increments
            transform (torchvision.transforms, optional): A composition of torchvision transforms. Defaults to None.
        """
        self.path = dataset.path
        self.task = dataset.task
        self.mask = dataset.mask
        self.domain_incremental_task = domain_incremental_task
        self.domain_increments = domain_increments
        if dataset.raw_labels is not None:
            self.raw_labelset = dataset.raw_labelset
        self.raw_labels = dataset.raw_labels
        self.ratios = dataset.ratios
        self.split = split
        if transform is None:
            self.transform = lambda x: x
        else:
            self.transform = transform
        self.split_data(dataset.x, dataset.y, dataset.mask, dataset.ratios, dataset.domain_y)

    def split_data(self, x, y, mask, ratios, domain_y, rng=np.random.RandomState(42)):
        if mask is None:
            if self.split == 'train':
                start = 0
                end = int(ratios[0] * len(x))
            elif self.split == 'val':
                start = int(ratios[0] * len(x))
                end = int((ratios[0] + ratios[1]) * len(x))
            elif self.split == 'test':
                start = int((ratios[0] + ratios[1]) * len(x))
                end = len(x)
            indices = rng.permutation(len(x))
            indices = indices[start:end]
        else:
            mask = mask[:, ["train", "val", "test"].index(self.split)]
            indices = np.arange(len(y))  # 0....nsamples
            indices = indices[mask]
        self.labelset = list(sorted(set(y)))
        self.y = np.array([self.labelset.index(y) for y in y])
        self.x = x[indices]
        self.y = self.y[indices]
        if self.raw_labels is not None:
            self.raw_labels = np.array(self.raw_labels)[indices]

        # Create "domain_increments" bins from the attribute used for domain incremental learning 
        if self.domain_increments is not None:
            if self.domain_incremental_task in Synbols.categorical_attributes:
                self.domains = list(sorted(set(domain_y)))
                self.tasks_per_domain = len(self.domains) // self.domain_increments
                self.domain_labels = list(range(self.domain_increments)) * self.tasks_per_domain + list(range(self.domain_increments))[:(len(self.domains) % self.domain_increments)]
                self.task_id = domain_y[indices]
                self.task_id = np.array([self.domain_labels[self.domains.index(d_y)] for d_y in self.task_id])
            elif self.domain_incremental_task in Synbols.continuous_attributes:
                self.domains = np.linspace(domain_y.min(), domain_y.max() + 1e-4, self.domain_increments + 1)
                domain_y = domain_y[indices]
                self.task_id = np.zeros(len(domain_y), dtype=int)
                for i in range(1, self.domain_increments):
                    self.task_id[(domain_y >= self.domains[i - 1]) & (domain_y < self.domains[i])] = i
            else:
                raise ValueError("Domain attribute not found")
            
            if len(set(self.task_id)) != self.domain_increments:
                raise RuntimeError("""The number of tasks differs from the number of domain increments. 
                                        This could happen if the number of domains in the dataset is less than 
                                        the requested number of increments. For instance, requesting 2000 
                                        font increments for a dataset with 1500 fonts will result in failure""")
        else:
            self.task_id = None

    def __getitem__(self, item):
        if self.raw_labels is None:
            return self.transform(self.x[item]), self.y[item]
        else:
            return self.transform(self.x[item]), self.y[item], self.raw_labels[item]

    def __len__(self):
        return len(self.x)


def get_data_path_or_download(dataset, data_root):
    """Finds a dataset locally and downloads if needed.

    Args:
        dataset (str): dataset name. For instance 'camouflage_n=100000_2020-Oct-19.h5py'.
            See https://github.com/ElementAI/synbols-resources/tree/master/datasets/generated for the complete list. (please ignore .a[a-z] extensions)
        data_root (str): path where the dataset will be or is stored. If empty string, it defaults to $TMPDIR 

    Raises:
        ValueError: dataset name does not exist in local path nor in remote

    Returns:
        str: dataset final path 
    """
    url_prefix = 'https://github.com/ElementAI/synbols-resources/raw/master/datasets/generated/'
    if data_root == "":
        data_root = os.environ.get("TMPDIR", "/tmp")
    full_path = os.path.join(data_root, dataset)

    if os.path.isfile(full_path):
        print("%s found." % full_path)
        return full_path
    else:
        print("Downloading %s..." % full_path)

    r = requests.head(os.path.join(url_prefix, dataset))
    is_big = not r.ok

    if is_big:
        r = requests.head(os.path.join(url_prefix, dataset + ".aa"))
        if not r.ok:
            raise ValueError("Dataset %s" % dataset, "not found in remote.")
        response = input("Download more than 3GB (Y/N)?: ").lower()
        while response not in ["y", "n"]:
            response = input("Download more than 3GB (Y/N)?: ").lower()
        if response == "n":
            print("Aborted")
            sys.exit(0)
        parts = []
        current_part = "a"
        while r.ok:
            r = requests.head(os.path.join(
                url_prefix, dataset + ".a%s" % current_part))
            parts.append(".a" + current_part)
            current_part = chr(ord(current_part) + 1)
    else:
        parts = [""]

    if not os.path.isfile(full_path):
        with open(full_path, 'wb') as file:
            for i, part in enumerate(parts):
                print("Downloading part %d/%d" % (i + 1, len(parts)))
                url = os.path.join(url_prefix, "%s%s" % (dataset, part))

                # Streaming, so we can iterate over the response.
                response = requests.get(url, stream=True)
                total_size_in_bytes = int(
                    response.headers.get('content-length', 0))
                block_size = 1024  # 1 Kilobyte
                # progress_bar = tqdm(total=total_size_in_bytes,
                                    # unit='iB', unit_scale=True)
                for data in response.iter_content(block_size):
                    # progress_bar.update(len(data))
                    file.write(data)
                # progress_bar.close()
                # if total_size_in_bytes != 0:# and progress_bar.n != total_size_in_bytes:
                    # print("ERROR, something went wrong downloading %s" % url)
    return full_path