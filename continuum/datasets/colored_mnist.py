import numpy as np

from continuum.datasets import MNIST


class ColoredMNIST(MNIST):
    """Colored MNIST dataset.

    References:
    * Invariant Risk Minimization
      Arjovsky et al.
      arXiv 2019

    The dataset is made of two labels: 0 & 1.
    All the original digits from 0 to 4 are now 0, and the leftover are now 1.

    25% of the labels have been randomly flipped.

    Each label (0 & 1) has an assigned color (red or green). But flip_color% of
    the samples have their color flipped. If the model learns the spurrious correlation
    of the color, then it'll get super bad.

    Numpy version of the DomainBed's implementation:
    - https://github.com/facebookresearch/DomainBed/blob/main/domainbed/datasets.py#L126

    :param args: Parameters given to the MNIST dataset.
    :param flip_color: % of labels whose colors have been flipped. Original
                       paper used 0.1, 0.2, and 0.9.
    :param kargs: Named parameters given to the MNIST dataset.
    """
    def __init__(
        self,
        *args,
        flip_color: float = 0.9,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.flip_color = flip_color

    def get_data(self):
        x, y, t = super().get_data()

        images = x
        # Flip label with probability 0.25
        labels = (y < 5).astype(np.float32)

        # Flip label with probability 0.25
        labels = self._xor(labels,
                           self._bernoulli(0.25, len(labels)))

        # Assign a color based on the label; flip the color with probability e
        colors = self._xor(
            labels,
            self._bernoulli(self.flip_color, len(labels))
        )
        images = np.stack([images, images], axis=1)
        # Apply the color to the image by zeroing out the other color channel
        images[np.arange(len(images)), (
            1 - colors).astype(np.int64), :, :] *= 0

        images = np.concatenate(
            [images, np.zeros((images.shape[0], 1, images.shape[2], images.shape[3]))], axis=1)

        images = images.transpose(0, 2, 3, 1)
        return images, labels.astype(np.int64), t

    def _bernoulli(self, p, size):
        return (np.random.rand(size) < p).astype(np.float32)

    def _xor(self, a, b):
        return np.abs(a - b)
