# Continuum

[![PyPI version](https://badge.fury.io/py/continuum.svg)](https://badge.fury.io/py/continuum) [![Build Status](https://travis-ci.com/Continvvm/continuum.svg?branch=master)](https://travis-ci.com/Continvvm/continuum) [![Codacy Badge](https://api.codacy.com/project/badge/Grade/c3a31475bebc4036a13e6048c24eb3e0)](https://www.codacy.com/gh/Continvvm/continuum?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=Continvvm/continuum&amp;utm_campaign=Badge_Grade) [![DOI](https://zenodo.org/badge/254864913.svg)](https://zenodo.org/badge/latestdoi/254864913) [![Documentation Status](https://readthedocs.org/projects/continuum/badge/?version=latest)](https://continuum.readthedocs.io/en/latest/?badge=latest)

## A library for PyTorch's loading of datasets in the field of Continual Learning

Aka Continual Learning, Lifelong-Learning, Incremental Learning, etc.

Read the [documentation](https://continuum.readthedocs.io/en/latest/).

### Example:

Install from and PyPi:
```bash
pip3 install continuum
```

And run!
```python
from torch.utils.data import DataLoader

from continuum import ClassIncremental, split_train_val
from continuum.datasets import MNIST

clloader = ClassIncremental(
    MNIST("my/data/path", download=True),
    increment=1,
    initial_increment=5,
    train=True  # a different loader for test
)

print(f"Number of classes: {clloader.nb_classes}.")
print(f"Number of tasks: {clloader.nb_tasks}.")

for task_id, train_dataset in enumerate(clloader):
    train_dataset, val_dataset = split_train_val(train_dataset, val_split=0.1)
    train_loader = DataLoader(train_dataset)
    val_loader = DataLoader(val_dataset)

    for x, y, t in train_loader:
        # Do your cool stuff here
```

### Supported Scenarios

|Name | Acronym | Supported |
|:----|:---|:---:|
| **New Instances** | NI | :white_check_mark: |
| **New Classes** | NC | :white_check_mark: |
| **New Instances & Classes** | NIC | :white_check_mark: |

### Supported Datasets:

Note that the task sizes are fully customizable.

|Name | Nb classes | Image Size | Automatic Download | Type |
|:----|:---:|:----:|:---:|:---:|
| **MNIST** | 10 | 28x28x1 | :white_check_mark: | :eyes: |
| **Fashion MNIST** | 10 | 28x28x1 | :white_check_mark: | :eyes: |
| **KMNIST** | 10 | 28x28x1 | :white_check_mark: | :eyes: |
| **EMNIST** | 10 | 28x28x1 | :white_check_mark: | :eyes: |
| **QMNIST** | 10 | 28x28x1 | :white_check_mark: | :eyes: |
| **MNIST Fellowship** | 30 | 28x28x1 | :white_check_mark: | :eyes: |
| **CIFAR10** | 10 | 32x32x3 | :white_check_mark: | :eyes: |
| **CIFAR100** | 100 | 32x32x3 | :white_check_mark: | :eyes: |
| **CIFAR Fellowship** | 110 | 32x32x3 | :white_check_mark: | :eyes: |
| **ImageNet100** | 100 | 224x224x3 | :x: | :eyes: |
| **ImageNet1000** | 1000 | 224x224x3 | :x: | :eyes: |
| **Permuted MNIST** | 10 | 28x28x1 | :white_check_mark: | :eyes: |
| **Rotated MNIST** | 10 | 28x28x1 | :white_check_mark: | :eyes: |
| **CORe50** | 50 | 224x224x3 | :white_check_mark: | :eyes: |
| **CORe50-v2-79** | 50 | 224x224x3 | :white_check_mark: | :eyes: |
| **CORe50-v2-196** | 50 | 224x224x3 | :white_check_mark: | :eyes: |
| **CORe50-v2-391** | 50 | 224x224x3 | :white_check_mark: | :eyes: |
| **MultiNLI** | 5 | | :white_check_mark: | :book: |


Furthermore some "Meta"-datasets are available:

**InMemoryDataset**, for in-memory numpy array:
```python
x_train, y_train = gen_numpy_array()
x_test, y_test = gen_numpy_array()

clloader = CLLoader(
    InMemoryDataset(x_train, y_train, x_test, y_test),
    increment=10,
)
```

**PyTorchDataset**,for any dataset defined in torchvision:
```python
clloader = CLLoader(
    PyTorchDataset("/my/data/path", dataset_type=torchvision.datasets.CIFAR10),
    increment=10,
)
```

**ImageFolderDataset**, for datasets having a tree-like structure, with one folder per class:
```python
clloader = CLLoader(
    ImageFolderDataset("/my/train/folder", "/my/test/folder"),
    increment=10,
)
```

**Fellowship**, to combine several continual datasets.:
```python
clloader = CLLoader(
    Fellowship("/my/data/path", dataset_list=[CIFAR10, CIFAR100]),
    increment=10,
)
```

Some datasets cannot provide an automatic download of the data for miscealleneous reasons. For example for ImageNet, you'll need to download the data from the [official page](http://www.image-net.org/challenges/LSVRC/2012/downloads). Then load it likewise:
```python
clloader = CLLoader(
    ImageNet1000("/my/train/folder", "/my/test/folder"),
    increment=10,
)
```

Some papers use a subset, called ImageNet100 or ImageNetSubset. They are automatically
downloaded for you, but you can also provide your own.


### Continual Loader

#### Class Incremental

The Continual Loader `ClassIncremental` loads the data and batch it in several
tasks, each with new classes. See there some example arguments:

```python
from continuum import ClassIncremental

clloader = ClassIncremental(
    my_continual_dataset,
    increment=10,
    initial_increment=2,
    train_transformations=[transforms.RandomHorizontalFlip()],
    common_transformations=[
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ],
    train=True
)
```

Here the first task is made of 2 classes, then all following tasks of 10 classes. You can have a more finegrained increment by providing a list of `increment=[2, 10, 5, 10]`.

The `train_transformations` is applied only on the training data, while the `common_transformations` on both the training and testing data.

If you want a clloader for the test data, you'll need to create a new instance with `train=False`.

#### Instance Incremental

Tasks can also be made of new instances. By default the samples images are randomly
shuffled in different tasks, but some datasets provide, in addition of the data `x` and labels `y`,
a task id `t` per sample. For example `MultiNLI`, a NLP dataset, has 5 classes but
with 10 different domains. Each domain represents a new task.

```python
from continuum import InstanceIncremental
from continuum.datasets import MultiNLI

clloader = InstanceIncremental(
    MultiNLI("/my/path/where/to/download"),
    train=True
)
```

#### New Class & Instance

NIC settting is a special case of NI setting. For now, only the CORe50 dataset
supports this setting.

### Indexing

All our continual loader are iterable (i.e. you can for loop on them), and are
also indexable.

Meaning that `clloader[2]` returns the third task (index starts at 0). Likewise,
if you want to evaluate after each task, on all seen tasks do `clloader_test[:n]`.

### Sample Images

**MNIST**:

|<img src="images/mnist_0.jpg" width="150">|<img src="images/mnist_1.jpg" width="150">|<img src="images/mnist_2.jpg" width="150">|<img src="images/mnist_3.jpg" width="150">|<img src="images/mnist_4.jpg" width="150">|
|:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:|
|Task 0 | Task 1 | Task 2 | Task 3 | Task 4|

**FashionMNIST**:

|<img src="images/fashion_mnist_0.jpg" width="150">|<img src="images/fashion_mnist_1.jpg" width="150">|<img src="images/fashion_mnist_2.jpg" width="150">|<img src="images/fashion_mnist_3.jpg" width="150">|<img src="images/fashion_mnist_4.jpg" width="150">|
|:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:|
|Task 0 | Task 1 | Task 2 | Task 3 | Task 4|

**CIFAR10**:

|<img src="images/cifar10_0.jpg" width="150">|<img src="images/cifar10_1.jpg" width="150">|<img src="images/cifar10_2.jpg" width="150">|<img src="images/cifar10_3.jpg" width="150">|<img src="images/cifar10_4.jpg" width="150">|
|:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:|
|Task 0 | Task 1 | Task 2 | Task 3 | Task 4|

**MNIST Fellowship (MNIST + FashionMNIST + KMNIST)**:

|<img src="images/mnist_fellowship_0.jpg" width="150">|<img src="images/mnist_fellowship_1.jpg" width="150">|<img src="images/mnist_fellowship_2.jpg" width="150">|
|:-------------------------:|:-------------------------:|:-------------------------:|
|Task 0 | Task 1 | Task 2 |


**PermutedMNIST**:

|<img src="images/mnist_permuted_0.jpg" width="150">|<img src="images/mnist_permuted_1.jpg" width="150">|<img src="images/mnist_permuted_2.jpg" width="150">|<img src="images/mnist_permuted_3.jpg" width="150">|<img src="images/mnist_permuted_4.jpg" width="150">|
|:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:|
|Task 0 | Task 1 | Task 2 | Task 3 | Task 4|

**RotatedMNIST**:

|<img src="images/mnist_rotated_0.jpg" width="150">|<img src="images/mnist_rotated_1.jpg" width="150">|<img src="images/mnist_rotated_2.jpg" width="150">|<img src="images/mnist_rotated_3.jpg" width="150">|<img src="images/mnist_rotated_4.jpg" width="150">|
|:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:|
|Task 0 | Task 1 | Task 2 | Task 3 | Task 4|


**ImageNet100**:

|<img src="images/imagenet100_0.jpg" width="150">|<img src="images/imagenet100_1.jpg" width="150">|<img src="images/imagenet100_2.jpg" width="150">|<img src="images/imagenet100_3.jpg" width="150">| ... |
|:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:|
|Task 0 | Task 1 | Task 2 | Task 3 | ... |


### Citation

If you find this library useful in your work, please consider citing it:

```
@misc{douillardlesort2020continuum,
  author={Douillard, Arthur and Lesort, Timothée},
  title={Continuum, Data Loaders for Continual Learning},
  howpublished={https://github.com/Continvvm/continuum},
  year={2020},
  doi={10.5281/zenodo.3759673}
}
```

### Maintainers

This project was started by a joint effort from [Arthur Douillard](https://arthurdouillard.com/) &
[Timothée Lesort](https://tlesort.github.io/).

Feel free to contribute! If you want to propose new features, please create an issue.


### On PyPi

Our project is available on PyPi!

```bash
pip3 install continuum
```

Note that previously another project, a CI tool, was using that name. It is now
there [continuum_ci](https://pypi.org/project/continuum_ci/).
