import numpy as np
from clloader.datasets import MNIST
from scipy import ndimage


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

    def init(self):
        base_train, base_test = MNIST.init(self)

        x_train, y_train = [base_train[0]], [base_train[1]]
        x_test, y_test = [base_test[0]], [base_test[1]]

        class_increment = len(np.unique(base_train[1]))
        for i, value in enumerate(self._transformations, start=1):
            trsf_train, trsf_test = self._transform(base_train[0], base_test[0], value)

            x_train.append(trsf_train)
            x_test.append(trsf_test)

            y_train.append(base_train[1] + i * class_increment)
            y_test.append(base_test[1] + i * class_increment)

        x_train = np.concatenate(x_train)
        y_train = np.concatenate(y_train)
        x_test = np.concatenate(x_test)
        y_test = np.concatenate(y_test)

        return (x_train, y_train), (x_test, y_test)

    def _transform(self, x_train, x_test, seed):
        random_state = np.random.RandomState(seed=seed)
        permutations = random_state.permutation(x_train.shape[1] * x_train.shape[2])

        train_shape = x_train.shape
        test_shape = x_test.shape

        x_train = x_train.reshape((train_shape[0], -1))[..., permutations].reshape(train_shape)
        x_test = x_test.reshape((test_shape[0], -1))[..., permutations].reshape(test_shape)

        return x_train, x_test


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
        MNIST.__init__(self, *args, **kwargs)

        self._transformations = angles
        self._mapping = None

    def _transform(self, x_train, x_test, angle):
        x_train = ndimage.rotate(x_train, angle=angle, axes=(2, 1), reshape=False)
        x_test = ndimage.rotate(x_test, angle=angle, axes=(2, 1), reshape=False)

        return x_train, x_test
