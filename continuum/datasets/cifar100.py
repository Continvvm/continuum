
from torchvision import datasets as torchdata
from torchvision import transforms
from continuum.datasets import PyTorchDataset

import numpy as np

class CIFAR100(PyTorchDataset):

    def __init__(self, classification: str = "object", *args, **kwargs):
        super().__init__(*args, dataset_type=torchdata.cifar.CIFAR100, **kwargs)
        assert classification in ["object", "category"]

        if self.classification == "category":
            # Classes labels from 0-19 (instead of 0 to 99)
            # cf : "superclasses" or "coarse labels" https://www.cs.toronto.edu/~kriz/cifar.html

            # code from https://github.com/ryanchankh/cifar100coarse/blob/master/cifar100coarse.py
            # update labels
            coarse_labels = np.array([4, 1, 14, 8, 0, 6, 7, 7, 18, 3,
                                      3, 14, 9, 18, 7, 11, 3, 9, 7, 11,
                                      6, 11, 5, 10, 7, 6, 13, 15, 3, 15,
                                      0, 11, 1, 10, 12, 14, 16, 9, 11, 5,
                                      5, 19, 8, 8, 15, 13, 14, 17, 18, 10,
                                      16, 4, 17, 4, 2, 0, 17, 4, 18, 17,
                                      10, 3, 2, 12, 12, 16, 12, 1, 9, 19,
                                      2, 10, 0, 1, 16, 12, 9, 13, 15, 13,
                                      16, 19, 2, 4, 6, 19, 5, 5, 8, 19,
                                      18, 1, 2, 15, 6, 0, 17, 8, 14, 13])
            self.targets = coarse_labels[self.targets]

            # update classes
            self.classes = [['beaver', 'dolphin', 'otter', 'seal', 'whale'],
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



    @property
    def transformations(self):
        return [
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        ]
