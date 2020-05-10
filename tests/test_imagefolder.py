import os
import tempfile

import numpy as np
import pytest
from PIL import Image
from torch.utils.data import DataLoader

from continuum.datasets import ImageFolderDataset
from continuum.scenarios import ClassIncremental


def gen_imagefolder(path, nb_classes=10, nb_samples=2):
    for class_id in range(nb_classes):
        folder = os.path.join(path, f"class_{class_id}")
        os.makedirs(folder)

        for sample_id in range(nb_samples):
            arr = np.random.randint(0, 255, size=(32, 32, 3))
            img = Image.fromarray(arr.astype("uint8"))
            img.save(os.path.join(folder, f"sample_{sample_id}.jpg"))


# yapf: disable

@pytest.mark.parametrize("increment,initial_increment,nb_tasks", [
    (2, 0, 5),
    (5, 0, 2),
    (1, 5, 6),
    (2, 4, 4),
    ([5, 1, 1, 3], 0, 4)
])
def test_increments(increment, initial_increment, nb_tasks):
    with tempfile.TemporaryDirectory() as train_path, tempfile.TemporaryDirectory() as test_path:
        gen_imagefolder(train_path)
        gen_imagefolder(test_path)

        clloader = ClassIncremental(
            ImageFolderDataset(train_path, test_path),
            increment=increment,
            initial_increment=initial_increment
        )

        assert clloader.nb_tasks == nb_tasks
        seen_tasks = 0
        for task_id, train_dataset in enumerate(clloader):
            seen_tasks += 1

            if isinstance(increment, list):
                max_class = sum(increment[:task_id + 1])
                min_class = sum(increment[:task_id])
            elif initial_increment:
                max_class = initial_increment + increment * task_id
                min_class = initial_increment + increment * (task_id -1) if task_id > 0 else 0
            else:
                max_class = increment * (task_id + 1)
                min_class = increment * task_id

            for _ in DataLoader(train_dataset):
                pass

            assert np.max(train_dataset._y) == max_class - 1
            assert np.min(train_dataset._y) == min_class
    assert seen_tasks == nb_tasks
