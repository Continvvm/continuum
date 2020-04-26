from torchvision import datasets as torchdata
from torchvision import transforms

from clloader.datasets import PyTorchDataset


class CIFAR10(PyTorchDataset):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, dataset_type=torchdata.cifar.CIFAR10, **kwargs)

    @property
    def transformations(self):
        return [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ]


class CIFAR100(PyTorchDataset):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, dataset_type=torchdata.cifar.CIFAR100, **kwargs)

    @property
    def transformations(self):
        return [
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        ]


class MNIST(PyTorchDataset):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, dataset_type=torchdata.MNIST, **kwargs)


class FashionMNIST(PyTorchDataset):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, dataset_type=torchdata.FashionMNIST, **kwargs)


class KMNIST(PyTorchDataset):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, dataset_type=torchdata.KMNIST, **kwargs)


class EMNIST(PyTorchDataset):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, dataset_type=torchdata.EMNIST, **kwargs)


class QMNIST(PyTorchDataset):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, dataset_type=torchdata.QMNIST, **kwargs)
