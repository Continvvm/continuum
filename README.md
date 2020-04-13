# Continual Loader (CLLoader)

[![PyPI version](https://badge.fury.io/py/clloader.svg)](https://badge.fury.io/py/clloader) [![Build Status](https://travis-ci.com/arthurdouillard/continual_loader.svg?branch=master)](https://travis-ci.com/arthurdouillard/continual_loader)

## A library for PyTorch's loading of datasets in the field of Continual Learning

Aka Continual Learning, Lifelong-Learning, Incremental Learning, etc.

### Example:

```python
from torch.utils.data import DataLoader

from clloader import CLLoader
from clloader.datasets import MNIST

clloader = CLLoader(
    MNIST("my/data/path", download=True),
    increment=1,
    initial_increment=5
)

print(f"Number of classes: {clloader.nb_classes}.")
print(f"Number of tasks: {clloader.nb_tasks}.")

for task_id, (train_dataset, test_dataset) in enumerate(clloader):
    train_loader = DataLoader(train_dataset)
    test_loader = DataLoader(test_dataset)

    # Do your cool stuff here
```

### Supported Scenarios

|Name | Acronym | Supported |
|:----|:---|:---:|
| **New Instances** | NI | :x: |
| **New Classes** | NC | :white_check_mark: |
| **New Instances & Classes** | NIC | :x: |

### Supported Datasets:

Note that the task sizes are fully customizable.

|Name | Nb classes | Image Size | Automatic Download |
|:----|:---:|:----:|:---:|
| **MNIST** | 10 | 28x28x1 | :white_check_mark: |
| **Fashion MNIST** | 10 | 28x28x1 | :white_check_mark: |
| **KMNIST** | 10 | 28x28x1 | :white_check_mark: |
| **EMNIST** | 10 | 28x28x1 | :white_check_mark: |
| **QMNIST** | 10 | 28x28x1 | :white_check_mark: |
| **MNIST Fellowship** | 30 | 28x28x1 | :white_check_mark: |
| **CIFAR10** | 10 | 32x32x3 | :white_check_mark: |
| **CIFAR100** | 100 | 32x32x3 | :white_check_mark: |
| **CIFAR Fellowship** | 110 | 32x32x3 | :white_check_mark: |
| **ImageNet100** | 100 | 224x224x3 | :x: |
| **ImageNet1000** | 1000 | 224x224x3 | :x: |
| **Permuted MNIST** | 10 + X * 10 | 224x224x3 | :white_check_mark: |

Furthermore some "Meta"-datasets are available:
- **InMemoryDataset**: for in-memory numpy array
- **PyTorchDataset**: for any dataset defined in torchvision
- **ImageFolderDataset**: for datasets having a tree-like structure, with one folder per class
- **Fellowship**: to combine several datasets

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
