from typing import Tuple, Union, Optional, List

import numpy as np
import torch
from torchvision import transforms

from continuum.tasks.base import BaseTaskSet, TaskType

try:
    import soundfile
except:
    soundfile = None


class AudioTaskSet(BaseTaskSet):
    """A task dataset specific to  text returned by the CLLoader.

    :param x: The data, text here
    :param y: The targets, not one-hot encoded.
    :param t: The task id of each sample.
    :param trsf: The transformations to apply on the images.
    :param target_trsf: The transformations to apply on the labels.
    """

    def __init__(
            self,
            x: np.ndarray,
            y: np.ndarray,
            t: np.ndarray,
            trsf: Union[transforms.Compose, List[transforms.Compose]],
            target_trsf: Optional[Union[transforms.Compose, List[transforms.Compose]]] = None
    ):
        if not soundfile:
            raise ImportError("You need to install the soundfile library to work on audio data.")

        super().__init__(x, y, t, trsf, target_trsf)
        self.data_type = TaskType.AUDIO

    def plot(
            self,
            path: Union[str, None] = None,
            title: str = "",
            nb_samples: int = 100,
            shape: Optional[Tuple[int, int]] = None,
    ) -> None:
        """Plot samples of the current task, useful to check if everything is ok.

        :param path: If not None, save on disk at this path.
        :param title: The title of the figure.
        :param nb_samples: Amount of samples randomly selected.
        :param shape: Shape to resize the image before plotting.
        """
        raise NotImplementedError("we do not plot audio task set yet.")

    def get_sample(self, index: int) -> np.ndarray:
        """Returns a sample data corresponding to the given `index`.

        :param index: Index to query the image.
        :return: the sample data.
        """
        return torch.FloatTensor(soundfile.read(self._x[index])[0])

    def __getitem__(self, index: int) -> Tuple[np.ndarray, int, int]:
        """Method used by PyTorch's DataLoaders to query a sample and its target."""
        x = self.get_sample(index)
        y = self._y[index]
        t = self._t[index]

        trsf = self.get_task_trsf(t)
        if trsf:
            x = trsf(x)

        if self.target_trsf is not None:
            y = self.get_task_target_trsf(t)(y)

        return x, y, t



