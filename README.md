# Continual Loader (CLLoader)

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

### Supported Datasets:

- MNIST
- FashionMNIST
- KMNIST
- EMNIST
- QMNIST
- CIFAR10
- CIFAR100
- ImageNet100
- ImageNet1000
- MNIST Fellowship (MNIST + FashionMNIST + EMNIST)
- CIFAR Fellowship (CIFAR10 + CIFAR100)

Furthermore some "Meta"-datasets are available:
- InMemoryDataset: for in-memory numpy array
- PyTorchDataset: for datasets defined in torchvision
- ImageFolderDataset: for datasets having a tree-like structure, with one folder per class
- Fellowship: to combine several datasets
