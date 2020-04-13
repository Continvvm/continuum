import abc
from typing import List, Tuple, Union

import numpy as np

from clloader.datasets import MNIST
from torchvision import datasets as torchdata
from torchvision import transforms


class PermutedMNIST(MNIST):

    def __init__(self, nb_permutations=4, **kwargs):
        super().__init__(**kwargs)

        self.nb_permutations = nb_permutations

    def init(self):
        base_train, base_test = super().init()

        x_train, y_train = [base_train[0]], [base_train[1]]
        x_test, y_test = [base_test[0]], [base_test[1]]

        class_counter = np.max(base_train[1])
        nb_base_class = class_counter
        for i in range(self.nb_permuations):
            permuted_train, permuted_test = self._permut(base_train[0], base_test[0], i)

            x_train.append(permuted_train)
            x_test.append(permuted_test)

            y_train.append(base_train[1] + class_counter)
            y_test.append(base_test[1] + class_counter)

            class_counter += nb_base_class

        x_train = np.concatenate(x_train)
        y_train = np.concatenate(y_train)
        x_test = np.concatenate(x_test)
        y_test = np.concatenate(y_test)

        return (x_train, y_train), (x_test, y_test)

    def _permut(self, x_train, x_test, i):
        random_state = np.random.random_state(seed=i)
        permutations = random_state.permutation(x_train.shape[1] * x_train.shape[2])

        train_shape = x_train.shape
        test_shape = x_test.shape

        x_train = x_train.reshape((train_shape[0], -1))[..., permutations].reshape(train_shape)
        x_test = x_test.reshape((test_shape[0], -1))[..., permutations].reshape(test_shape)

        return x_train, x_test
