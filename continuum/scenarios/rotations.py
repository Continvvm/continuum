from typing import Callable, List, Tuple, Union

from continuum.datasets import _ContinuumDataset
from continuum.scenarios import TransformationIncremental

from torchvision import transforms


class Rotations(TransformationIncremental):
    """Continual Loader, generating datasets for the consecutive tasks.

    Scenario: Rotations scenario is a new instance scenario.
              For each task data is rotated from a certain angle.

    :param cl_dataset: A continual dataset.
    :param nb_tasks: The scenario's number of tasks.
    :param list_degrees: list of rotation in degree (int) or list of range. e.g. (0, (40,45), 90).
    :param base_transformations: Preprocessing transformation to applied to data before rotation.
    """

    def __init__(
            self,
            cl_dataset: _ContinuumDataset,
            nb_tasks: int,
            list_degrees: Union[List[Tuple], List[int]],
            base_transformations: List[Callable] = None
    ):

        if len(list_degrees) != nb_tasks:
            raise ValueError("The nb of tasks != with number of angles tuples set in the list")

        list_transformations = []
        min, max = None, None
        for tuple_ in list_degrees:
            if isinstance(tuple_, int):
                min = tuple_
                max = tuple_
            elif len(tuple_) == 2:
                min, max = tuple_
            else:
                # list_degrees should contains list of rotations with:
                # only one angle
                # or with (min,max)
                raise ValueError("list_degrees is wrong")

            list_transformations.append([transforms.RandomAffine(degrees=[min, max])])

        super(Rotations, self).__init__(
            cl_dataset=cl_dataset,
            nb_tasks=nb_tasks,
            incremental_transformations=list_transformations,
            base_transformations=base_transformations
        )
