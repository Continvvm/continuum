"""A collection of internal utilities. Not made for user!"""
from typing import Optional, List

import numpy as np



def _slice(
    y: np.ndarray,
    t: Optional[np.ndarray],
    keep_classes: Optional[List[int]] = None,
    discard_classes: Optional[List[int]] = None,
    keep_tasks: Optional[List[int]] = None,
    discard_tasks: Optional[List[int]] = None
):
    """Slice dataset to keep/discard some classes/task-ids.

    Note that keep_* and and discard_* are mutually exclusive.
    Note also that if a selection (keep or discard) is being made on the classes
    and on the task ids, the resulting intersection will be taken.

    :param y: An array of class ids.
    :param t: An array of task ids.
    :param keep_classes: Only keep samples with these classes.
    :param discard_classes: Discard samples with these classes.
    :param keep_tasks: Only keep samples with these task ids.
    :param discard_tasks: Discard samples with these task ids.
    :return: A new Continuum dataset ready to be given to a scenario.
    """
    if keep_classes is not None and discard_classes is not None:
        raise ValueError("Only use `keep_classes` or `discard_classes`, not both.")
    if keep_tasks is not None and discard_tasks is not None:
        raise ValueError("Only use `keep_tasks` or `discard_tasks`, not both.")

    if t is None and (keep_tasks is not None or discard_tasks is not None):
        raise Exception(
            "No task ids information is present by default with this dataset, "
            "thus you cannot slice some task ids."
        )
    y, t = y.astype(np.int64), t.astype(np.int64)

    indexes = set()
    if keep_classes:
        indexes = set(np.where(np.isin(y, keep_classes))[0])
    elif discard_classes:
        keep_classes = list(set(y) - set(discard_classes))
        indexes = set(np.where(np.isin(y, keep_classes))[0])

    if keep_tasks:
        _indexes = np.where(np.isin(t, keep_tasks))[0]
        if len(indexes) > 0:
            indexes = indexes.intersection(_indexes)
        else:
            indexes = indexes.union(_indexes)
    elif discard_tasks:
        keep_tasks = list(set(t) - set(discard_tasks))
        _indexes = np.where(np.isin(t, keep_tasks))[0]
        if len(indexes) > 0:
            indexes = indexes.intersection(_indexes)
        else:
            indexes = indexes.union(_indexes)

    indexes = np.array(list(indexes), dtype=np.int64)
    return indexes
