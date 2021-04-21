from continuum.datasets import Fellowship, CIFAR10, SVHN, FashionMNIST, MNIST, DTD

# TODO: when re-viewing the same dataset, we'll sample the same dataset. Is that a problem?

class CTRLplus(Fellowship):  # S^-
    def __init__(self, data_path: str = "", train: bool = True, download: bool = True, seed: int = 1):
        datasets = [
            CIFAR10(data_path=data_path, train=train, download=download),
            MNIST(data_path=data_path, train=train, download=download),
            DTD(data_path=data_path, train=train, download=download),
            FashionMNIST(data_path=data_path, train=train, download=download),
            SVHN(data_path=data_path, train=train, download=download),
            CIFAR10(data_path=data_path, train=train, download=download)
        ]

        if train:
            proportions = [4000, 400, 400, 400, 400, 400]
        else:
            proportions = [2000, 200, 200, 200, 200, 200]

        super().__init__(
            datasets=datasets,
            update_labels=True,
            proportions=proportions,
            seed=seed
        )


class CTRLminus(Fellowship):  # S^+
    def __init__(self, data_path: str = "", train: bool = True, download: bool = True, seed: int = 1):
        datasets = [
            CIFAR10(data_path=data_path, train=train, download=download),
            MNIST(data_path=data_path, train=train, download=download),
            DTD(data_path=data_path, train=train, download=download),
            FashionMNIST(data_path=data_path, train=train, download=download),
            SVHN(data_path=data_path, train=train, download=download),
            CIFAR10(data_path=data_path, train=train, download=download)
        ]

        if train:
            proportions = [400, 400, 400, 400, 400, 4000]
        else:
            proportions = [200, 200, 200, 200, 200, 2000]

        super().__init__(
            datasets=datasets,
            update_labels=True,
            proportions=proportions,
            seed=seed
        )


class CTRLin(Fellowship):  # S^{in}
    def __init__(self, data_path: str = "", train: bool = True, download: bool = True, seed: int = 1):
        datasets = [
            CIFAR10(data_path=data_path, train=train, download=download),
            MNIST(data_path=data_path, train=train, download=download),
            DTD(data_path=data_path, train=train, download=download),
            FashionMNIST(data_path=data_path, train=train, download=download),
            SVHN(data_path=data_path, train=train, download=download),

            # TODO: need modified rainbow mnist there
            # TODO: need different input distribution
        ]

        if train:
            proportions = [4000, 400, 400, 400, 400, 400]
        else:
            proportions = [2000, 200, 200, 200, 200, 200]

        super().__init__(
            datasets=datasets,
            update_labels=True,
            proportions=proportions,
            seed=seed
        )


class CTRLout(Fellowship):  # S^{out}
    def __init__(self, data_path: str = "", train: bool = True, download: bool = True, seed: int = 1):
        datasets = [
            CIFAR10(data_path=data_path, train=train, download=download),
            MNIST(data_path=data_path, train=train, download=download),
            DTD(data_path=data_path, train=train, download=download),
            FashionMNIST(data_path=data_path, train=train, download=download),
            SVHN(data_path=data_path, train=train, download=download),

            # TODO: need different output distribution
            CIFAR10(data_path=data_path, train=train, download=download),
        ]

        if train:
            proportions = [4000, 400, 400, 400, 400, 400]
        else:
            proportions = [2000, 200, 200, 200, 200, 200]

        super().__init__(
            datasets=datasets,
            update_labels=True,
            proportions=proportions,
            seed=seed
        )


class CTRLpl(Fellowship):  # S^{pl}
    def __init__(self, data_path: str = "", train: bool = True, download: bool = True, seed: int = 1):
        datasets = [
            MNIST(data_path=data_path, train=train, download=download),
            DTD(data_path=data_path, train=train, download=download),
            FashionMNIST(data_path=data_path, train=train, download=download),
            SVHN(data_path=data_path, train=train, download=download),
            CIFAR10(data_path=data_path, train=train, download=download)
        ]

        if train:
            proportions = [400, 400, 400, 400, 400, 4000]
        else:
            proportions = [200, 200, 200, 200, 200, 2000]

        super().__init__(
            datasets=datasets,
            update_labels=True,
            proportions=proportions,
            seed=seed
        )


# TODO: add CTRLlong
