import pickle as pkl
import os
from typing import Iterable, Tuple, Union
from random import randint, seed

import numpy as np

from continuum import download
from continuum.datasets.base import _ContinuumDataset
from continuum.tasks import TaskType

class MetaShift(_ContinuumDataset):
    # NOTE : using random_contexts can cause some contexts to contain few examples.
    """Continuum version of the Core50 dataset.

    References:
        * MetaShift: A Dataset of Datasets for Evaluating Contextual 
          Distribution Shifts and Training Conflicts
          Weixin Liang, James Zou
          2022
          arXiv:2202.06523v1

    :param data_path: The folder path containing the data.
    :param train : 
    :param download: If true, will download images and the dictionnary 
           containing classes and contexts if not already in data_path folder.
    :param train_image_ids: Images ids to use.
    :param class_names : classes to consider (default = None --> all classes)
    :param random_context : if true ; only one occurence of each image in a 
           random choosen class&context combination.
    :param random_seed : set seed (relevant ponly if random context is True)
    :param nb_task : set the number of distict tasks.
    """
    data_url = "https://nlp.stanford.edu/data/gqa/images.zip"
    pickle_url = "https://github.com/Weixin-Liang/MetaShift/blob/main/dataset/meta_data/full-candidate-subsets.pkl?raw=true"

    def __init__(
            self,
            data_path:str,
            train:bool = True,
            download:bool = True,
            class_names:Union[Iterable[str], None] = None, #Only get images corresponding to specific classe(s)
            train_image_ids:Union[Iterable[str], None] = None, #Only get specific ids for training. (if None : all)
            random_contexts:bool = False, #If true, the images will be assigned random class(context) combination among all valid combinations.
            random_seed:int = 42,
            nb_tasks:int = 0
            ):

        self.train_image_ids = train_image_ids
        self.random_contexts = random_contexts
        self.class_names = class_names
        self.seed = random_seed
        self.nb_tasks = nb_tasks
        super().__init__(data_path, train, download)

    @property
    def data_type(self) -> TaskType:
        return TaskType.IMAGE_PATH

    def _download(self):
        # Visual Genome Dataset
        if os.path.exists(os.path.join(self.data_path, "images")):
            print("Dataset already extracted.")
        else:
            path = download.download(self.data_url, self.data_path)
            download.unzip(path)
            print("Dataset extracted.")

        # Pickle file
        if os.path.exists(os.path.join(self.data_path, "full-candidate-subsets.pkl")):
            print("Classes and contexts already downloaded")
        else:
            file = download.download(self.pickle_url, self.data_path)
            os.rename(file, os.path.join(self.data_path, "full-candidate-subsets.pkl"))
            print("Classes and contexts downloaded")
    
    def get_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate Metashift Data
        Metashift classifies the Visual Genome dataset into classes and contexts. 
        Images can be found in many classes and contexts. random_context is False, 
        images will appear many times within the return value. We thus might get :

          x       y       t

        1.jpg   class1  context1
        1.jpg   class1  context2
        1.jpg   class2  context1
        1.jpg   class2  context3
        1.jpg   class2  context4
        2.jpg   class3  context2
        ...     ...     ...

        x contains all images ids present in at least 1 context.
        y contains the class of the image.
        t contains the context in which the image is represented.
        """
        x, y, t = [], [], []

        with open(os.path.join(self.data_path, "full-candidate-subsets.pkl"), 'rb') as pkl_file:
            pkl_dict = pkl.load(pkl_file)
        

        for key in pkl_dict.keys():
                # Iterate through all class(context) pairs
                
            context_name = key.split("(")[1][:-1]
            class_name = key.split("(")[0]

            if (self.class_names is None or class_name in self.class_names) :
                
                for id in pkl_dict[key]:    # Iterate through all ids

                    if self.train_image_ids is None or id in self.train_image_ids :
                        x.append(os.path.join(self.data_path, "images", "images", id)+".jpg")
                        y.append(class_name)
                        t.append(context_name)


        x, y, t = np.array(x), np.array(y), np.array(t)
        

        if(self.random_contexts == True):
            x, y, t = _select_random_contexts(x, y, t, self.seed)

        y = np.unique(y, return_inverse = True)[1]
        n, t = np.unique(t, return_inverse = True)
        n = len(n)

        if (self.nb_tasks > 0) and (n > self.nb_tasks): # to be tested
            scale = lambda x : x % self.nb_tasks # scale task ids to number of tasks
            t = scale(t)

        return x, y, t

def _select_random_contexts(x, y, t, rand_seed): 
    #choose a random context for each class(context) combiation for each image amoung available.    
    idx_sort = np.argsort(x)
    sorted_x = x[idx_sort]
    sorted_y = y[idx_sort]
    sorted_t = t[idx_sort]

    final_idx_list = []

    vals, idx_start, count = np.unique(sorted_x, return_counts=True, return_index=True)

    seed(rand_seed)

    for i in range(len(vals)):
        final_idx_list.append(randint(idx_start[i], idx_start[i]+count[i]-1))
        
    x2 = sorted_x[final_idx_list]
    y2 = sorted_y[final_idx_list]
    t2 = sorted_t[final_idx_list]

    return x2, y2, t2