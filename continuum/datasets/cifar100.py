from typing import Tuple

import numpy as np
from torchvision import datasets as torchdata
from torchvision import transforms

from continuum.datasets import PyTorchDataset


cifar100_coarse_labels = np.array([4, 1, 14, 8, 0, 6, 7, 7, 18, 3,
                                   3, 14, 9, 18, 7, 11, 3, 9, 7, 11,
                                   6, 11, 5, 10, 7, 6, 13, 15, 3, 15,
                                   0, 11, 1, 10, 12, 14, 16, 9, 11, 5,
                                   5, 19, 8, 8, 15, 13, 14, 17, 18, 10,
                                   16, 4, 17, 4, 2, 0, 17, 4, 18, 17,
                                   10, 3, 2, 12, 12, 16, 12, 1, 9, 19,
                                   2, 10, 0, 1, 16, 12, 9, 13, 15, 13,
                                   16, 19, 2, 4, 6, 19, 5, 5, 8, 19,
                                   18, 1, 2, 15, 6, 0, 17, 8, 14, 13])


def get_lifelong_cifar100(y):
    """"
    Create a task label such as having one of coarse label in each task but one 20 different classes in each task.
    """

    # they should be 20 corase labels with 5 classes each
    coarse_labels = np.unique(cifar100_coarse_labels)
    assert len(coarse_labels) == 20

    # we should have a 5*20 matrix with indexes of classes of each coarse_labels
    np_indexes_coarse_labels = np.zeros((5, 20))
    for i in range(20):
        indexes_coarse_labels = np.where(cifar100_coarse_labels == i)[0]
        assert len(indexes_coarse_labels) == 5, print(f"len(indexes_coarse_labels): {len(indexes_coarse_labels)}")
        np_indexes_coarse_labels[:, i] = np.array(indexes_coarse_labels)

    np_indexes_coarse_labels = np_indexes_coarse_labels.astype(int)

    # we can have 5 tasks with 1 classes per coarse labels to make a lifelong scenario

    # so we create the task label vector
    t = np.zeros(len(y))
    for i in range(5):
        indexes = np_indexes_coarse_labels[i,:]
        assert len(np.unique(cifar100_coarse_labels[indexes])) == 20, print(cifar100_coarse_labels[indexes])
        assert len(indexes) == 20, print(f"len(indexes) {len(indexes)}")
        for index in indexes:
            data_index_class = np.where(y==index)[0]
            t[data_index_class] = i

    return t.astype(int)

class CIFAR100(PyTorchDataset):
    """Continuum use of the CIFAR100 dataset.
    We adapt this dataset to create scenario from the combination of both of category labels and class labels.

    References:
        * "Learning Multiple Layers of Features from Tiny Images"
          Alex Krizhevsky

    :param data_path: The folder path containing the data.
    :param download: An option useless in this case.
    :param labels_type: labels type define if we use class labels or category labels for each data point.
    :param task_labels: labels type define what type of labels we use if we want to create a task id vector.
    """

    def __init__(self, *args, labels_type: str = "class", task_labels: str = None, **kwargs):
        super().__init__(*args, dataset_type=torchdata.cifar.CIFAR100, **kwargs)
        if not labels_type in ["class", "category"]:
            AssertionError("unknown labels_type parameter, choose among ['class', 'category']")
        if not  task_labels in [None, "class", "category"]:
            AssertionError("unknown task_labels parameter, choose among [None, 'class', 'category', 'lifelong']")

        self.labels_type = labels_type
        self.task_labels_type = task_labels

        # lifelong scenario can not be with 'class' label_type
        if task_labels == 'lifelong':
            labels_type = 'category'

        if self.task_labels_type is None:
            # the dataset does not provide a task id vector
            self.t = None
        elif self.task_labels_type == "class":
            # the dataset provides a task id vector based on classes
            self.t = self.dataset.targets
        elif self.task_labels_type == "category":
            # the dataset provides a task id vector based on categories
            self.t = cifar100_coarse_labels[self.dataset.targets]
        elif self.task_labels_type == "lifelong":
            self.t = get_lifelong_cifar100(self.dataset.targets)

        if self.labels_type == "category":
            # here we replace class labels by category labels to annotate data.

            # Classes labels from 0-19 (instead of 0 to 99)
            # cf : "superclasses" or "coarse labels" https://www.cs.toronto.edu/~kriz/cifar.html

            # code from https://github.com/ryanchankh/cifar100coarse/blob/master/cifar100coarse.py
            self.dataset.targets = cifar100_coarse_labels[self.dataset.targets]

            # update classes
            self.dataset.classes = [['beaver', 'dolphin', 'otter', 'seal', 'whale'],
                                    ['aquarium_fish', 'flatfish', 'ray', 'shark', 'trout'],
                                    ['orchid', 'poppy', 'rose', 'sunflower', 'tulip'],
                                    ['bottle', 'bowl', 'can', 'cup', 'plate'],
                                    ['apple', 'mushroom', 'orange', 'pear', 'sweet_pepper'],
                                    ['clock', 'keyboard', 'lamp', 'telephone', 'television'],
                                    ['bed', 'chair', 'couch', 'table', 'wardrobe'],
                                    ['bee', 'beetle', 'butterfly', 'caterpillar', 'cockroach'],
                                    ['bear', 'leopard', 'lion', 'tiger', 'wolf'],
                                    ['bridge', 'castle', 'house', 'road', 'skyscraper'],
                                    ['cloud', 'forest', 'mountain', 'plain', 'sea'],
                                    ['camel', 'cattle', 'chimpanzee', 'elephant', 'kangaroo'],
                                    ['fox', 'porcupine', 'possum', 'raccoon', 'skunk'],
                                    ['crab', 'lobster', 'snail', 'spider', 'worm'],
                                    ['baby', 'boy', 'girl', 'man', 'woman'],
                                    ['crocodile', 'dinosaur', 'lizard', 'snake', 'turtle'],
                                    ['hamster', 'mouse', 'rabbit', 'shrew', 'squirrel'],
                                    ['maple_tree', 'oak_tree', 'palm_tree', 'pine_tree', 'willow_tree'],
                                    ['bicycle', 'bus', 'motorcycle', 'pickup_truck', 'train'],
                                    ['lawn_mower', 'rocket', 'streetcar', 'tank', 'tractor']]

    def get_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        x, y, _ = super().get_data()
        return x, y, self.t

    @property
    def transformations(self):
        """Default transformations if nothing is provided to the scenario."""
        return [transforms.ToTensor(),
                transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))]
