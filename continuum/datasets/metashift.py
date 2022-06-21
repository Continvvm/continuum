import pickle as pkl
import os
from typing import Iterable, Tuple, Union
import numpy as np
from continuum import download
from continuum.datasets.base import _ContinuumDataset
from continuum.tasks import TaskType
from warnings import warn
import zipfile

class MetaShift(_ContinuumDataset):
    # NOTE : using unique_occurence can cause some contexts to contain few examples.
    """Continuum version of the MetaShift dataset.

    References:
        * MetaShift: A Dataset of Datasets for Evaluating Contextual 
          Distribution Shifts and Training Conflicts
          Weixin Liang, James Zou
          2022
          arXiv:2202.06523v1

    :param data_path: The folder path containing the data.
    :param train : (bool) Default is True.
    :param download: If true, will download images and the dictionnary 
           containing classes and contexts if not already in data_path folder.
    :param class_names : classes to consider (default = None --> all classes)
    :param context_names : ccontexts to consider (default = None --> all contexts)
    :param unique_occurence : If true ; only one occurence of each image in a 
           random choosen class&context combination. If false, all valid contexts are considered, 
           duplicates of ids will be found in x.
    :param random_seed : set seed (relevant only if random context is True)
    :param nb_task : set the number of distict tasks.
    :param strict_domain_inc : If true, only contexts represented in all classes are kept. If false, all contexts are kept. (default = false)
    """
    data_url = "https://nlp.stanford.edu/data/gqa/images.zip"
    pickle_url = "https://github.com/Weixin-Liang/MetaShift/blob/main/dataset/meta_data/full-candidate-subsets.pkl?raw=true"
    targets_url = "https://nlp.stanford.edu/data/gqa/sceneGraphs.zip"

    def __init__(
            self,
            data_path:str,
            visual_genome_path:Union[str, None] = None,
            train:bool = True,
            download:bool = True,
            class_names:Union[Iterable[str], None] = None, #Only get images corresponding to specific classe(s)
            context_names:Union[Iterable[str], None] = None, #Only get images corresponding to specific context(s)
            unique_occurence:bool = False, #If true, the images will be assigned random class(context) combination among all valid combinations.
            random_seed:int = 42,
            nb_tasks:int = 0,
            strict_domain_inc:bool = False
            ):

        self.unique_occurence = unique_occurence
        self.visual_genome_path = os.path.expanduser(visual_genome_path) if visual_genome_path is not None else os.path.join(os.path.expanduser(data_path), "MetaShift", "images")
        self.class_names = class_names
        self.context_names = context_names
        self.seed = random_seed
        self.nb_tasks = nb_tasks
        self.strict_domain_inc = strict_domain_inc
        self.class_names_updated = None
        self.context_names_updated = None
        super().__init__(data_path, train, download)

    @property
    def data_type(self) -> TaskType:
        return TaskType.IMAGE_PATH

    def _download(self):
        # Visual Genome Dataset
        if os.path.exists(self.visual_genome_path):
            print("Dataset already extracted.")
        else:
            print("BEWARE : 20GB file is downloading")
            path = os.path.join(self.data_path, "MetaShift")
            if not os.path.exists(path) : os.mkdir(path)
            file = download.download(self.data_url, self.data_path)
            with zipfile.ZipFile(file, 'r') as zip_file:
                zip_file.extractall(path)
            print("\nDataset extracted.")

        # Pickle file
        if os.path.exists(os.path.join(self.data_path, "full-candidate-subsets.pkl")):
            print("Classes and contexts already downloaded")
        else:
            file = download.download(self.pickle_url, self.data_path)
            os.rename(file, os.path.join(self.data_path, "full-candidate-subsets.pkl"))
            print("\nClasses and contexts downloaded")
        
        # Visual Genome targets
        if os.path.exists(os.path.join(self.data_path, "sceneGraphs")):
            print("Visual Genome targets already downloaded")
        else:
            file = download.download(self.targets_url, self.data_path)
            path = os.path.join(self.data_path, "sceneGraphs")
            if not os.path.exists(path) : os.mkdir(path)
            with zipfile.ZipFile(file, 'r') as zip_file:
                zip_file.extractall(path)
            print("\nVisual Genome targets downloaded")
    
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

            if ( (self.class_names is None or class_name in self.class_names) and (self.context_names is None or context_name in self.context_names) ):
                
                for id in pkl_dict[key]:    # Iterate through all ids
                    x.append(os.path.join(self.visual_genome_path, id)+".jpg")
                    y.append(class_name)
                    t.append(context_name)

        x, y, t = np.array(x), np.array(y), np.array(t)

        if(self.unique_occurence == True):
            x, y, t = self._select_unique_occurence(x, y, t)

        self.class_names_updated, y = np.unique(y, return_inverse = True)

        if (self.strict_domain_inc == True):
            x, y, t = self._strict_domain_tasks(x, y, t)
            self.context_names_updated, t = np.unique(t, return_inverse=True)
        else:
            self.context_names_updated, t = np.unique(t, return_inverse = True) 
        
        n = np.max(t) + 1

        if (self.nb_tasks > 0) and (n > self.nb_tasks): # to be tested
            t = t % self.nb_tasks # scale task ids to number of tasks

        return x, y, t

    def get_class_context_in_order(self):
        """Returns arrays of class names and context names such that the indices correspond to the y and t values. (If nb_task was not fixed.)
        If the get_data() method was not called or no scenario was created, warns the user and returns (None, None).

        Returns:
            tuple[NDArray | None, NDArray | None]: array of class names, array of context names
        """
        if self.class_names_updated is None or self.context_names_updated is None:
            warn("get_class_context_in_order, the class and task indices are not known yet : get_data() was not called and no scenario was created.")
        
        return self.class_names_updated, self.context_names_updated
    
    def _strict_domain_tasks(self, x, y, t):
        # selects tasks for which all classes are represented.
        unique_tasks, t2 = np.unique(t, return_inverse=True)

        selected_tasks = np.zeros([len(unique_tasks),], dtype=np.int8)

        # for each task, verify if the number of classes in task corresponds to the number of classes.
        for i, context in enumerate(unique_tasks):
            classes = np.unique(y[np.where(t==context)])
            if len(classes) == len(self.class_names):
                selected_tasks[i] = 1
        
        # If no tasks contains all classes : raise an error
        if np.all(selected_tasks == 0):
            raise ValueError("Error : No task contains all classes. Try with fewer classes or set strict_domain_inc to false.")

        # Create subset for selected tasks.
        idx_selected = np.where(selected_tasks[t2]==1)

        return x[idx_selected], y[idx_selected], t[idx_selected]


    def _select_unique_occurence(self, x, y, t): 
        # choose an unique context for each data point (randomly)    
        np.random.seed(seed=self.seed)

        # shuffle
        permutation = np.random.permutation(len(x))
        x2, y2, t2 = x[permutation], y[permutation], t[permutation]

        # keep first occurence
        x2, idx = np.unique(x2, return_index=True)
        y2, t2 = y2[idx], t2[idx]

        return x2, y2, t2

def get_all_classes_contexts(data_path):
    """Returns list of all class names and context names present in the original MetaShift dataset.

    Args:
        data_path (str): path to data folder containing full-candidate-subsets.pkl

    Returns:
        tuple[List, List] : List of class names, List of context names
    """
    with open(os.path.join(data_path, "full-candidate-subsets.pkl"), 'rb') as pkl_file:
        pkl_dict = pkl.load(pkl_file)
    
    all_classes = []
    all_contexts = []
    for key in pkl_dict.keys():
        class_name, context = key.split('(')
        context = context[:-1]
        if class_name not in all_classes:
            all_classes.append(class_name)
        if context not in all_contexts:
            all_contexts.append(context)

    return all_classes, all_contexts

def get_all_contexts_from_classes(data_path, class_names):
    """Returns a list of all context names corresponding to given class names in MetaShift dataset.

    Args:
        data_path (str): path to folder containing full-candidate-subsets.pkl
        class_names (List): class names

    Returns:
        List : context names corresponding to class names.
    """
    with open(os.path.join(data_path, "full-candidate-subsets.pkl"), 'rb') as pkl_file:
        pkl_dict = pkl.load(pkl_file)
    
    all_contexts = []
    for key in pkl_dict.keys():
        class_name, context = key.split('(')
        context = context[:-1]
        if class_name in class_names and context not in all_contexts:
            all_contexts.append(context)
    return all_contexts

