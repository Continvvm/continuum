import os

import pytest
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader

from continuum.datasets import InMemoryDataset
from continuum.scenarios import SegmentationClassIncremental


@pytest.fixture
def pseudo_voc(tmpdir):
    nb_samples = 20

    x = np.random.randint(0, 255, (nb_samples, 2, 2, 3), dtype=np.uint8)
    y = np.zeros((nb_samples, 2, 2), dtype=np.uint8)
    y[0:15, 0, 0] = 255

    y[0:10, 0, 1] = 1
    y[4:10, 1, 0] = 2
    y[5:20, 0, 1] = 3
    y[15:20, 1, 1] = 4

    x_paths, y_paths = [], []
    for i in range(nb_samples):
        x_paths.append(os.path.join(tmpdir, f"seg_{i}.jpg"))
        y_paths.append(os.path.join(tmpdir, f"seg_{i}.png"))

        Image.fromarray(x[i]).save(x_paths[-1])
        Image.fromarray(y[i]).save(y_paths[-1])

    return InMemoryDataset(
        np.array(x_paths), np.array(y_paths),
        data_type="segmentation"
    )


@pytest.fixture
def pseudo_voc_png(tmpdir):
    nb_samples = 20

    x = np.zeros((nb_samples, 2, 2, 3), dtype=np.uint8)
    y = np.zeros((nb_samples, 2, 2), dtype=np.uint8)
    y[0:15, 0, 0] = 255

    y[0:10, 0, 1] = 1
    y[4:10, 1, 0] = 2
    y[5:20, 0, 1] = 3
    y[15:20, 1, 1] = 4

    x[0:10, 0, 1] = 1
    x[4:10, 1, 0] = 2
    x[5:20, 0, 1] = 3
    x[15:20, 1, 1] = 4

    x = np.tile(y[..., None], (1, 1, 1, 3))

    x_paths, y_paths = [], []
    for i in range(nb_samples):
        x_paths.append(os.path.join(tmpdir, f"seg2_{i}.png"))
        y_paths.append(os.path.join(tmpdir, f"seg2_{i}.png"))

        Image.fromarray(x[i]).save(x_paths[-1])
        Image.fromarray(y[i]).save(y_paths[-1])

    return InMemoryDataset(
        np.array(x_paths), np.array(y_paths),
        data_type="segmentation"
    )


@pytest.mark.parametrize("mode,lengths,increment", [
    ("overlap", (10, 15), 2),
    ("disjoint", (5, 15), 2),
    ("disjoint", (5, 10, 5), 1),
    ("overlap", (10, 15, 5), 1),
    ("sequential", (5, 15), 2),
    ("sequential", (5, 10, 5), 1),
])
def test_length_taskset(pseudo_voc, mode, lengths, increment):
    scenario = SegmentationClassIncremental(
        pseudo_voc,
        nb_classes=4,
        increment=increment,
        initial_increment=2,
        mode=mode
    )

    assert len(scenario) == len(lengths)
    for i, l in enumerate(lengths):
        assert len(scenario[i]) == l, i


@pytest.mark.parametrize("mode,class_order,error", [
    ("overlap", [1, 2, 3, 4], False),
    ("overlap", [1, 2, 3, 4, 5], True),
    ("overlap", [0, 1, 2, 3], True),
    ("overlap", [1, 2, 3, 255], True),
    ("disjoint", [1, 2, 3, 4], False),
    ("overlap", [4, 3, 2, 1], False),
    ("overlap", [2, 3, 4, 1], False),

])
def test_class_order(pseudo_voc_png, mode, class_order, error):
    """We need PNG here because JPG lose some pixels values"""
    increments = [1, 1, 1, 1]

    if error:
        with pytest.raises(ValueError):
            scenario = SegmentationClassIncremental(
                pseudo_voc_png,
                nb_classes=4,
                increment=increments,
                class_order=class_order,
                mode=mode
            )
        return
    else:
        scenario = SegmentationClassIncremental(
            pseudo_voc_png,
            nb_classes=4,
            increment=increments,
            class_order=class_order,
            mode=mode
        )

    for task_id, task_set in enumerate(scenario):
        loader = DataLoader(task_set, batch_size=200, drop_last=False)
        x, y, _ = next(iter(loader))
        pixels = torch.unique((x * 255).long())

        assert (task_id + 1) in y
        real_class = class_order[task_id]
        assert real_class in pixels, task_id
        original_y = scenario.get_original_targets(y)
        assert real_class in np.unique(original_y), task_id


@pytest.mark.parametrize("mode,increment", [
    ("overlap", 2),
    ("overlap", 1),
    ("disjoint", 2),
    ("disjoint", 1),
    ("sequential", 2),
    ("sequential", 1),
    ("overlap", [2, 2]),
    ("disjoint", [2, 2]),
    ("disjoint", [2, 1, 1]),
    ("overlap", [2, 1, 1]),
    ("disjoint", [1, 3]),
    ("overlap", [1, 3]),
    ("disjoint", [3, 1]),
    ("overlap", [3, 1]),
])
def test_labels(pseudo_voc, mode, increment):
    initial_increment = 2
    nb_classes = 4
    min_cls = 1

    scenario = SegmentationClassIncremental(
        pseudo_voc,
        nb_classes=nb_classes,
        increment=increment,
        initial_increment=initial_increment,
        mode=mode
    )

    if isinstance(increment, int) and increment == 2:
        increments = [2, 2]
    elif isinstance(increment, int) and increment == 1:
        increments = [2, 1, 1]
    else:
        increments = increment

    for task_id, task_set in enumerate(scenario):
        loader = DataLoader(task_set, batch_size=200, drop_last=False)
        x, y, t = next(iter(loader))

        assert len(x.shape) == 4
        assert len(y.shape) == 3
        assert len(t.shape) == 1
        assert x.shape[2:] == y.shape[1:]

        assert (t == task_id).all()

        seen_classes = set(torch.unique(y).numpy().tolist())

        max_cls = min_cls + increments[task_id]
        assert 0 in seen_classes, task_id
        if 4 not in seen_classes:
            assert 255 in seen_classes, task_id

        for c in list(range(min_cls, nb_classes + min_cls)):
            if mode in ("overlap", "disjoint"):
                if min_cls <= c < max_cls:
                    assert c in seen_classes, (c, task_id, min_cls, max_cls)
                else:
                    assert c not in seen_classes, (c, task_id, min_cls, max_cls)
            elif mode == "sequential":
                if c < max_cls:
                    assert c in seen_classes, (c, task_id, min_cls, max_cls)
                else:
                    assert c not in seen_classes, (c, task_id, min_cls, max_cls)

        min_cls += increments[task_id]
