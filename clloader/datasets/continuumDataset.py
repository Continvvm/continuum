
import torch
from clloader.datasets import BaseDataset
from clloader.viz import plot
from PIL import Image

class ContinuumDataset(TorchDataset):
    """A task dataset returned by the CLLoader.

    :param x: The data, either image-arrays or paths to images saved on disk.
    :param y: The targets, not one-hot encoded.
    :param trsf: The transformations to apply on the images.
    :param open_image: Whether to open image from disk, or index in-memory.
    """

    def __init__(
        self, x: np.ndarray, y: np.ndarray, trsf: transforms.Compose, open_image: bool = False
    ):
        self.x, self.y = x, y
        self.trsf = trsf
        self.open_image = open_image

    @property
    def nb_classes(self):
        """The number of classes contained in the current task."""
        return len(np.unique(self.y))

    def add_memory(self, x_memory: np.ndarray, y_memory: np.ndarray):
        """Add memory for rehearsal.

        :param x_memory: Sampled data chosen for rehearsal.
        :param y_memory: The associated targets of `x_memory`.
        """
        self.x = np.concatenate((self.x, x_memory))
        self.y = np.concatenate((self.y, y_memory))

    def plot(self, path=None, title="", nb_per_class=5, shape=None):
        """Plot samples of the current task, useful to check if everything is ok.

        :param path: If not None, save on disk at this path.
        :param title: The title of the figure.
        :param nb_per_class: Amount to sample per class.
        :param shape: Shape to resize the image before plotting.
        """
        plot(self, title=title, path=path, nb_per_class=nb_per_class, shape=shape)

    def __len__(self):
        """The amount of images in the current task."""
        return self.x.shape[0]

    def get_image(self, index):
        """Returns a Pillow image corresponding to the given `index`.

        :param index: Index to query the image.
        :return: A Pillow image.
        """
        x = self.x[index]
        if self.open_image:
            img = Image.open(x).convert("RGB")
        else:
            img = Image.fromarray(x.astype("uint8"))
        return img

    def __getitem__(self, index):
        """Method used by PyTorch's DataLoaders to query a sample and its target."""
        img = self.get_image(index)
        y = self.y[index]
        img = self.trsf(img)
        return img, y