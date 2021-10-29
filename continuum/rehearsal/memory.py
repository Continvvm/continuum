from typing import Union, Callable, Tuple, Any, Tuple, Optional, List
import warnings

import numpy as np

from continuum.rehearsal import herd_random, herd_closest_to_cluster, herd_closest_to_barycenter
from continuum import utils


class RehearsalMemory:
    """Class to handle the rehearsal data.

    # TODO: how does it handle non-class incremental setting?
    # TODO: what if new data contains old classes?

    :param memory_size: Total amount of memory (in number of samples) to allocate.
    :param herding_method: A string indicating how to herd, or a function. Please
                           see the predfined herding methods if you want to
                           create your own.
    :param fixed_memory: Fix the amount per class instead of using all available
                         memory capacity. This is usually fixed for paper starting
                         with half the classes like LUCIR (Hou et al. CVPR2019)
                         and PODNet (Douillard et al. ECCV2020).
    :param nb_total_classes: In case of fixed memory, precise the total amount of
                             classes that will be seen.
    """
    def __init__(
        self,
        memory_size: int,
        herding_method: Union[str, Callable],
        fixed_memory: bool = False,
        nb_total_classes: Union[None, int] = None
    ):
        if isinstance(herding_method, str):
            if herding_method == "random":
                herding_method = herd_random
            elif herding_method == "cluster":
                herding_method = herd_closest_to_cluster
            elif herding_method == "barycenter":
                herding_method = herd_closest_to_barycenter
            else:
                raise NotImplementedError(
                    f"Unknown rehearsal method {herding_method}, "
                    f"Provide a string ('random', 'cluster', or 'barycenter')"
                )
        elif hasattr(herding_method, "__call__"):
            pass
        else:
            raise NotImplementedError(
               f"Unknown rehearsal method {herding_method}, "
               f"Either provide its string name or a callable."
            )

        if fixed_memory and nb_total_classes is None:
            raise ValueError(
                f"When memory is fixed per class, you need to provide the `nb_total_classes` info."
            )

        self.memory_size = memory_size
        self.herding_method = herding_method
        self.fixed_memory = fixed_memory
        self.nb_total_classes = nb_total_classes
        self.seen_classes = set()

        self._x = self._y = self._t = None

    @property
    def nb_classes(self) -> int:
        """Current number of seen classes."""
        return len(self.seen_classes)

    @property
    def memory_per_class(self) -> int:
        """Number of samples per classes."""
        if self.fixed_memory:
            return self.memory_size // self.nb_total_classes
        elif self.nb_classes > 0:
            return self.memory_size // self.nb_classes
        return self.memory_size

    def save(self, path: str = "memory.npz"):
        """Save memory on disk in a single file."""
        np.savez(
            path,
            x=self._x, y=self._y, t=self._t
        )

    def load(self, path: str = "memory.npz"):
        """Load memory on disk from a single file.

        Warning! It'll erase existing memory!
        """
        if self._x is not None:
            warnings.warn(
                "Rehearsal memory wasn't empty when loading new memory from disk!"
                "Existing memory is erased to the profit of the loaded memory."
            )

        data = np.load(path)
        self._x = data["x"]
        self._y = data["y"]
        self._t = data["t"]

        self.seen_classes = set([y for y in np.unique(self._y)])

    def slice(
        self,
        keep_classes: Optional[List[int]] = None,
        discard_classes: Optional[List[int]] = None,
        keep_tasks: Optional[List[int]] = None,
        discard_tasks: Optional[List[int]] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Slice memory to keep/discard some classes/task-ids.

        Note that keep_* and and discard_* are mutually exclusive.
        Note also that if a selection (keep or discard) is being made on the classes
        and on the task ids, the resulting intersection will be taken.

        :param keep_classes: Only keep samples with these classes.
        :param discard_classes: Discard samples with these classes.
        :param keep_tasks: Only keep samples with these task ids.
        :param discard_tasks: Discard samples with these task ids.
        :return: The data x, y, t that has been sliced. Ready to be given to a new
                 memory object, an InMemoryDataset, or a TaskSet.
        """
        x, y, t = self.get()

        indexes = utils._slice(
            y, t,
            keep_classes, discard_classes,
            keep_tasks, discard_tasks
        )

        new_x, new_y, new_t = x[indexes], y[indexes], t[indexes]
        return new_x, new_y, new_t

    def get(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get the memory data to be added to a TaskSet."""
        if self._x is None:
            raise ValueError("Memory is empty!")
        return self._x, self._y, self._t

    def __len__(self) -> int:
        """Amount of samples stored in memory."""
        if self._x is None:
            return 0
        return len(self._x)

    def _reduce(self) -> None:
        """Reduce the amount of old classes when new classes are stored in memory."""
        x, y, t = [], [], []
        for class_id in np.unique(self._y):
            indexes = np.where(self._y == class_id)[0]
            x.append(self._x[indexes[:self.memory_per_class]])
            y.append(self._y[indexes[:self.memory_per_class]])
            t.append(self._t[indexes[:self.memory_per_class]])

        self._x = np.concatenate(x)
        self._y = np.concatenate(y)
        self._t = np.concatenate(t)

    def add(
        self,
        x: np.ndarray,
        y: np.ndarray,
        t: np.ndarray,
        z: Any
    ) -> None:
        """Add new classes to the memory.

        :param x: Input data (images, paths, etc.)
        :param y: Labels of the data.
        :param t: Task ids of the data.
        :param z: Extra info, could be many things but usually the extracted features.
        """
        for c in np.unique(y):
            self.seen_classes.add(c)

        mem_x, mem_y, mem_t = self.herding_method(x, y, t, z, nb_per_class=self.memory_per_class)

        if self._x is None:
            self._x, self._y, self._t = mem_x, mem_y, mem_t
        else:
            self._x = np.concatenate((self._x, mem_x))
            self._y = np.concatenate((self._y, mem_y))
            self._t = np.concatenate((self._t, mem_t))

        self._reduce()
