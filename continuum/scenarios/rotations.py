from typing import Callable, List, Tuple, Union

from torchvision import transforms

from continuum.datasets import _ContinuumDataset
from continuum.scenarios import TransformationIncremental


class Rotations(TransformationIncremental):
    """Continual Loader, generating datasets for the consecutive tasks.

    Scenario: Rotations scenario is a new instance scenario.
              For each task data is rotated from a certain angle.

    :param cl_dataset: A continual dataset.
    :param nb_tasks: The scenario's number of tasks.
    :param list_degrees: list of rotation in degree (int) or list of range. e.g. (0, (40,45), 90).
    :param base_transformations: Preprocessing transformation to applied to data before rotation.
    :param shared_label_space: If true same data with different transformation have same label
    """

    def __init__(
        self,
        cl_dataset: _ContinuumDataset,
        list_degrees: Union[List[Tuple], List[int]],
        nb_tasks: Union[int, None] = None,
        base_transformations: List[Callable] = None,
        shared_label_space=True
    ):

        if nb_tasks is not None and len(list_degrees) != nb_tasks:
            raise ValueError(
                f"The nb of tasks ({nb_tasks}) != number of angles "
                f"tuples ({len(list_degrees)}) set in the list"
            )

        trsfs = self._generate_transformations(list_degrees)

        super().__init__(
            cl_dataset=cl_dataset,
            incremental_transformations=trsfs,
            base_transformations=base_transformations,
            shared_label_space=shared_label_space
        )

    def _generate_transformations(self, degrees):
        trsfs = []
        min_deg, max_deg = None, None

        for deg in degrees:
            if isinstance(deg, int):
                min_deg, max_deg = deg, deg
            elif len(deg) == 2:
                min_deg, max_deg = deg
            else:
                raise ValueError(
                    f"Invalid list of degrees ({degrees}). "
                    "It should contain either integers (-deg, +deg) or "
                    "tuples (range) of integers (deg_a, deg_b)."
                )

            trsfs.append([transforms.RandomAffine(degrees=[min_deg, max_deg])])

        return trsfs
