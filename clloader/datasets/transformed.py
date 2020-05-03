from typing import Tuple

import numpy as np
from scipy import ndimage

from clloader.datasets import MNIST


class PermutedMNIST(MNIST):
    """A dataset made of MNIST and pixels permutations.

    The first task is the famous MNIST with 10 classes. Then all following tasks
    are the same MNIST but with pixels permuted in a random way.

    Note that classes are the same, only their representation changes.

    # Reference:
        * Overcoming catastrophic forgetting in neural networks
          Kirkpatrick et al.
          PNAS 2017

    :param nb_permutations: Number of permutations in addition of the original MNIST.
    """

    def __init__(self, *args, nb_permutations=4, **kwargs):
        MNIST.__init__(self, *args, **kwargs)

        self._transformations = list(range(nb_permutations))
        self._mapping = None

    @property
    def need_class_remapping(self) -> bool:
        """Flag for method `class_remapping`."""
        return True

    def class_remapping(self, class_ids: np.ndarray) -> np.ndarray:
        """Optional class remapping.

        Remap class ids so that whatever the permutations, the targets stay
        the same (0-9).
        For example, the second task, with permuted pixels, has targets (10-19)
        in order to mimick a class-incremental training but in reality those
        targets are (0-9).
        The remaping is done so that for the end-user, any tasks of PermutedMNIST
        has targets in the (0-9) range.

        :param class_ids: Original class_ids.
        :return: A remapping of the class ids.
        """
        if self._mapping is None:
            self._mapping = np.concatenate(
                [np.arange(10) for _ in range(len(self._transformations) + 1)]
            )
        return self._mapping[class_ids]

    def init(self, train: bool) -> Tuple[np.ndarray, np.ndarray, None]:
        base_data = MNIST.init(self, train)

        x, y = [base_data[0]], [base_data[1]]
        class_increment = len(np.unique(base_data[1]))

        for i, value in enumerate(self._transformations, start=1):
            x_transformed = self._transform(base_data[0], value)

            x.append(x_transformed)
            y.append(base_data[1] + i * class_increment)

        x = np.concatenate(x)
        y = np.concatenate(y)

        return x, y, None

    def _transform(self, x: np.ndarray, value: int) -> np.ndarray:
        # It's important to generate a new random state with a given seed
        # So that every run produces the same transformation,
        # and also that train & test have the same transformation.
        random_state = np.random.RandomState(seed=value)
        permutations = random_state.permutation(x.shape[1] * x.shape[2])

        shape = x.shape

        x_transformed = x.reshape((shape[0], -1))[..., permutations].reshape(shape)

        return x_transformed


class RotatedMNIST(PermutedMNIST):
    """A dataset made of MNIST and various rotations.

    The first task is the famous MNIST with 10 classes. Then all following tasks
    are the same MNIST but with a fixed rotations per task.

    Note that classes are the same, only their representation changes.

    # Reference:
        * Gradient Episodic Memory for Continual Learning
          Lopez-Paz and Ranzato
          NeurIPS 2017

    :param angles: A list of angles used in the rotation.
    """

    def __init__(self, *args, angles=[45, 90, 135, 180], **kwargs):
        MNIST.__init__(self, *args, **kwargs)  # pylint: disable=non-parent-init-called

        self._transformations = angles
        self._mapping = None

    def _transform(self, x: np.ndarray, value: int) -> np.ndarray:
        x_transformed = ndimage.rotate(x, angle=value, axes=(2, 1), reshape=False)
        return x_transformed
