from typing import Callable, List

import torch

from continuum.datasets import _ContinuumDataset
from continuum.scenarios import TransformationIncremental

from torchvision import transforms


class PermutationTransform:
    """Permutation transformers"""

    def __init__(self, seed):
        self.seed = seed
        self.g_cpu = torch.Generator()

    def __call__(self, x):
        shape = list(x.shape)
        x = x.view(-1)
        # if seed is 0, no permutations
        if self.seed != 0:
            self.g_cpu.manual_seed(self.seed)
            perm = torch.randperm(x.numel(), generator=self.g_cpu).long()
            x = x[perm]
        return x.view(shape)


class Permutations(TransformationIncremental):
    """Continual Loader, generating datasets for the consecutive tasks.
    Scenario: Mode incremental scenario is a new instance scenario where we explore the distribution mode by mode.
              For example rotation mnist, is a exploration of the distribution by rotation angles, each angle can be
              seen as a mode of the distribution. Same for permutMnist, mode=permutations space.

    :param cl_dataset: A continual dataset.
    :param nb_tasks: number of tasks.
    :param base_transformations: A list of data augmentation applied to the data set.
    :param seed: initialization seed for the permutations.
    """

    def __init__(
            self,
            cl_dataset: _ContinuumDataset,
            nb_tasks: int,
            base_transformations: List[Callable] = None,
            seed=0
    ):
        list_transformations = []
        g_cpu = torch.Generator()
        list_seed = torch.randperm(1000, generator=g_cpu)[:nb_tasks]
        self.seed = seed

        # first task is not permuted
        list_seed[0] = 0

        for seed in list_seed:
            list_transformations.append([PermutationTransform(seed.item())])

        super().__init__(
            cl_dataset=cl_dataset,
            nb_tasks=nb_tasks,
            incremental_transformations=list_transformations,
            base_transformations=base_transformations
        )

    # We inverse permutation is after self.trsf because it is done an torch tensor
    def get_task_transformation(self, task_index):
        return transforms.Compose(self.trsf.transforms + self.inc_trsf[task_index])
