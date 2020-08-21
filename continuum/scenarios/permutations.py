import warnings
from typing import Callable, List, Union

import numpy as np
import torch
from torchvision import transforms

from continuum.datasets import _ContinuumDataset
from continuum.scenarios import TransformationIncremental


class Permutations(TransformationIncremental):
    """Continual Loader, generating datasets for the consecutive tasks.

    Scenario: Permutations scenarios, use same data for all task but with pixels permuted.
    Each task get a specific permutation, such as all tasks are different.

    :param cl_dataset: A continual dataset.
    :param nb_tasks: The scenario's number of tasks.
    :param base_transformations: List of transformations to apply to all tasks.
    :param seed: initialization seed for the permutations.
    :param shared_label_space: If true same data with different transformation have same label
    """

    def __init__(
        self,
        cl_dataset: _ContinuumDataset,
        nb_tasks: Union[int, None] = None,
        base_transformations: List[Callable] = None,
        seed: Union[int, List[int]] = 0,
        shared_label_space=True
    ):
        trsfs = self._generate_transformations(seed, nb_tasks)

        super().__init__(
            cl_dataset=cl_dataset,
            incremental_transformations=trsfs,
            base_transformations=base_transformations,
            shared_label_space=shared_label_space
        )

    def _generate_transformations(self, seed, nb_tasks):
        if isinstance(seed, int):
            if nb_tasks is None:
                raise ValueError("You must specify a number of tasks if a single seed is provided.")
            rng = np.random.RandomState(seed=seed)
            seed = rng.permutation(100000)[:nb_tasks - 1]
        elif nb_tasks is not None and nb_tasks != len(seed) + 1:
            warnings.warn(
                f"Because a list of seed was provided {seed}, "
                f"the number of tasks is automatically set to "
                f"len(number of seeds) + 1 = {len(seed) + 1}"
            )

        return [PermutationTransform(seed=None)] + [PermutationTransform(seed=int(s)) for s in seed]

    def get_task_transformation(self, task_index):
        return transforms.Compose(self.trsf.transforms + [self.inc_trsf[task_index]])


class PermutationTransform:
    """Permutation transformers.

    This transformer is initialized with a seed such as same seed = same permutation.
    Seed 0 means no permutations

    :param seed: seed to initialize the random number generator
    """

    def __init__(self, seed: Union[int, None]):
        self.seed = seed
        self.g_cpu = torch.Generator()

    def __call__(self, x):
        shape = list(x.shape)
        x = x.reshape(-1)
        # if seed is None, no permutations
        if self.seed is not None:
            self.g_cpu.manual_seed(self.seed)
            perm = torch.randperm(x.numel(), generator=self.g_cpu).long()
            x = x[perm]
        return x.reshape(shape)
