import os

import pytest

from continuum.datasets import ImageNet100, ImageNet_R
from continuum.scenarios import ClassIncremental


nb_images_per_subset = {
    True: 129395,  # train
    False: 5000    # test
}

nb_images_per_subset_imagenet_r = {
    True: 24000,  # train
    False: 6000    # test
}
@pytest.fixture
def ImageNet100Test(tmpdir):
    folder = os.path.join(tmpdir, "imagenet100test")
    os.makedirs(folder)
    return ImageNet100(folder, data_subset=None, download=True, train=False)


@pytest.fixture
def ImageNet_RTrain(tmpdir):
    folder = os.path.join(tmpdir, "imagenet_rtrain")
    os.makedirs(folder)
    return ImageNet_R(folder, download=True, train=True)


@pytest.fixture
def ImageNet_RTest(tmpdir):
    folder = os.path.join(tmpdir, "imagenet_rtest")
    os.makedirs(folder)
    return ImageNet_R(folder, download=True, train=False)


@pytest.fixture
def ImageNet100Train(tmpdir):
    folder = os.path.join(tmpdir, "imagenet100train")
    os.makedirs(folder)
    return ImageNet100(folder, data_subset=None, download=True, train=True)


@pytest.mark.parametrize("train", [True, False])
def test_parsing_imagenet_r(ImageNet_RTrain, ImageNet_RTest, train):
    dataset = ImageNet_RTrain if train else ImageNet_RTest
    x, y, t = dataset.get_data()

    assert all("train" if train else "test" in path for path in x)

@pytest.mark.parametrize("train", [True, False])
def test_parsing_imagenet100(ImageNet100Train, ImageNet100Test, train):
    dataset = ImageNet100Train if train else ImageNet100Test
    x, y, t = dataset.get_data()

    assert all("train" if train else "test" in path for path in x)

@pytest.mark.parametrize("train", [True, False])
def test_nb_imagenet_r(ImageNet_RTrain, ImageNet_RTest, train):
    dataset = ImageNet_RTrain if train else ImageNet_RTest
    x, y, t = dataset.get_data()

    assert len(x) == nb_images_per_subset_imagenet_r[train]

@pytest.mark.parametrize("train", [True, False])
def test_nb_imagenet100(ImageNet100Train, ImageNet100Test, train):
    dataset = ImageNet100Train if train else ImageNet100Test
    x, y, t = dataset.get_data()

    assert len(x) == nb_images_per_subset[train]


# @pytest.mark.parametrize("train,div", [
#     (True, 1), (True, 2),
#     (False, 1), (True, 2)
# ])
# def test_customsubset_imagenet_r(ImageNet_RTrain, ImageNet_RTest, train, div):
#     dataset = ImageNet_RTrain if train else ImageNet_RTest
#     x, y, t = dataset.get_data()

#     new_x = x[:len(x) // div]
#     new_y = y[:len(y) // div]

#     subset = ImageNet_R(dataset.data_path, data_subset=(new_x, new_y), download=False, train=train)
#     x2, y2, t2 = subset.get_data()

#     assert len(x) // div == len(x2)



@pytest.mark.parametrize("train,div", [
    (True, 1), (True, 2),
    (False, 1), (True, 2)
])
def test_customsubset_imagenet100(ImageNet100Train, ImageNet100Test, train, div):
    dataset = ImageNet100Train if train else ImageNet100Test
    x, y, t = dataset.get_data()

    new_x = x[:len(x) // div]
    new_y = y[:len(y) // div]

    subset = ImageNet100(dataset.data_path, data_subset=(new_x, new_y), download=False, train=train)
    x2, y2, t2 = subset.get_data()

    assert len(x) // div == len(x2)



