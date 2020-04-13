import os
import tempfile

import numpy as np
import pytest
from PIL import Image

from clloader import CLLoader
from clloader.datasets import ImageFolderDataset
from torch.utils.data import DataLoader


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

        clloader = CLLoader(ImageFolderDataset(train_path, test_path), increment, initial_increment)

        assert clloader.nb_tasks == nb_tasks
        seen_tasks = 0
        for train_dataset, test_dataset in clloader:
            seen_tasks += 1

            for _ in DataLoader(train_dataset):
                pass
            for _ in DataLoader(test_dataset):
                pass

        assert seen_tasks == nb_tasks
