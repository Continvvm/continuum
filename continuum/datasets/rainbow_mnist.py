import numpy as np

from continuum.datasets import MNIST


class RainbowMNIST(MNIST):
    def __init__(
        self,
        *args,
        color: str = "red",
        rotation: float = 0.,
        scale=1.,
        **kwargs
    ):
        if rotation != 0. or scale != 1.:
            raise NotImplementedError(
                "Rotation and scale are not supported yet for RainbowMNIST."
            )
        if color not in ("red", "blue", "green"):
            raise ValueError(
                f"Only red, blue, and green colors are available, not {color}."
            )

        super().__init__(*args, **kwargs)

        self.color = color
        self.rotation = rotation
        self.scale = scale

    def get_data(self):
        x, y, t = super().get_data()
        x = x[..., None]

        if self.color == "red":
            c = np.zeros((x.shape[0], x.shape[1], x.shape[2], 2), dtype=np.uint8)
            x = np.concatenate((x, c), axis=-1)
        elif self.color == "green":
            c = np.zeros((x.shape[0], x.shape[1], x.shape[2], 1), dtype=np.uint8)
            x = np.concatenate((c, x, c), axis=-1)
        else:
            c = np.zeros((x.shape[0], x.shape[1], x.shape[2], 2), dtype=np.uint8)
            x = np.concatenate((c, x), axis=-1)

        return x, y, t
