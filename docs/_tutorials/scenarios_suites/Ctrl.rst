CTRL benchmark
--------------

CTRL is a collection of datasets proposed in
`"Efficient Continual Learning with Modular Networks and Task-Driven Priors" <https://arxiv.org/abs/2012.12631>`__ published at ICLR 2021
and co-authored by Veniat (Sorbonne), Denoyer and Ranzato (Facebook Research).

Each of the proposed dataset is combination of multiple small datasets among:

- MNIST
- SVHN
- CIFAR10
- Rainbow MNIST
- Fashion MNIST
- DTD

CTRL proposes different combinations which they name:

- CTRL minus
- CTRL plus
- CTRL in
- CTRL out
- CTRL plastic

The only way to use those datasets is with the `ContinualScenario` scenario, which is the most
flexible scenario Continuum offers.

You can a better feeling of this set of datasets in this
`Colab <https://colab.research.google.com/drive/1KNd2sJ9nG9h33hI4C3Ec7KDJmFCR5cdA?usp=sharing>`__ where
we display the images of each datasets and their statistics. Pay attention to the class labels!

Usage
=====

We can create a simple class incremental setting.

.. code-block:: python

    from continuum.datasets import CTRLminus, CTRLplus, CTRLplastic, CTRLin, CTRLout
    from continuum import ContinualScenario

    scenario_train = ContinualScenario(CTRLminus(path, split="train", download=True))
    scenario_train = ContinualScenario(CTRLminus(path, split="val", download=True))
    scenario_train = ContinualScenario(CTRLminus(path, split="test", download=True))

    for task_id, (train_set, val_set, test_set) in enumerate(zip(scenario_train, scenario_val, scenario_test)):
      train_loader = DataLoader(train_set)
      val_loader = DataLoader(val_set)
      test_loader = DataLoader(test_set)

      for x, y, t in train_loader:
        # Your model here


Note that contrarly to other datasets, all CTRL datasets have a split option which allow the use of
a particularly made validation set.

Custom CTRL
===========

Custom CTRL-like scenario (like the CTRL long that Veniat et al. describe) can
be generated programatically according to your rules. Beware that most are probably very
hard to solve but give it a try :)

.. code-block:: python

    from continuum.datasets import CTRL
    from continuum.datasets import MNIST, CIFAR10, FashionMNIST, SVHN


    class CTRLCustom(CTRL):
        def __init__(self, data_path: str = "", split: str = "train", download: bool = True, seed: int = 1):
            if split not in ("train", "val", "test"):
              raise ValueError(f"Split must be train, val, or test; not {split}.")
            train = split in ("train", "val")

            datasets = [
                CIFAR10(data_path=data_path, train=train, download=download),
                MNIST(data_path=data_path, train=train, download=download),
                FashionMNIST(data_path=data_path, train=train, download=download),
                SVHN(data_path=data_path, train=train, download=download),
                CIFAR10(data_path=data_path, train=train, download=download)
                CIFAR10(data_path=data_path, train=train, download=download)
            ]

            if split == "train":
                proportions = [4000, 400, 400, 400, 400, 400]
            elif split == "val":
                proportions = [2000, 200, 200, 200, 200, 200]
            else:
                proportions = None

            super().__init__(
                datasets=datasets,
                proportions=proportions,
                class_counter=[0, 10, 20, 30, 0, 40],
                class_subsets=[None, None, None, [0, 1, 2], None, None]
                seed=seed,
                split=split,
                target_size=(32, 32)
            )


What are the available customizations?

- **datasets**: you can choose any dataset you want, as a long as it's a
  `Continuum dataset <https://continuum.readthedocs.io/en/latest/_tutorials/datasets/dataset.html>`__.
  Beware that they will be loaded in memory (so avoid ImageNet datasets), and all resized to
  the **target_size**.
- **proportions**: it restricts the amount of data for train/val/test that will be used. Each class is
  sampled equally, therefore on CIFAR10, if I'm asking for 400 images, each class will have 40 images.
  If you don't want this option for a particular split, set it to `None`, as we do in the previous example.
  In this case, all split data will be used.
- **class_counter**: it controls what would be the i-th dataset labels. For example, if the `class_counter`
  is 30 for MNIST, then the dataset labels will be between 30 and 39. Thanks to this option, we can choose whether
  different datasets share the same labels. If you want MNIST and SVHN to share the same labels, they must have the
  same `class_counter`. In our code example, the 1-st and 5-th instances of CIFAR10 share the same labels, while
  the 6-th instance has different labels (although the actual classes are the same for a human).
- **class_subsets**: this options simply allows to select a subset of the classes. In the code example,
  we use all datasets classes except for SVHN where only the 0, 1, and 2 classes are used.

Now, if we wanted to generated complex random streams such as the CTRLlong of Veniat et al.,
we can combine those to generate a random stream like this:


.. code-block:: python

    import numpy as np

    class CTRLCustom(CTRL):
        def __init__(self, data_path: str = "", split: str = "train", download: bool = True, seed: int = 1):
            if split not in ("train", "val", "test"):
                raise ValueError(f"Split must be train, val, or test; not {split}.")
            train = split in ("train", "val")

            rng = np.random.RandomState(seed=seed)

            base_datasets = [
                MNIST(data_path=data_path, train=train, download=download),
                SVHN(data_path=data_path, train=train, download=download),
                FashionMNIST(data_path=data_path, train=train, download=download),
                CIFAR10(data_path=data_path, train=train, download=download)
            ]

            svhn_mnist_share_labels = True
            if svhn_mnist_share_labels:
                task_counter = [0, 0, 10, 10]
            else:
                task_counter = [0, 10, 20, 30]

            proportions_per_class = [1000, 1000, 1000, 500]
            dataset_sample_prob = [0.2, 0.2, 0.3, 0.3]
            nb_classes = 5
            nb_tasks = 30

            datasets, class_counter, class_subsets, proportions = [], [], [], []
            for _ in range(nb_tasks):
                dataset_id = rng.choice([0, 1, 2, 3], p=dataset_sample_prob)
                datasets.append(base_datasets[dataset_id])
                class_counter.append(task_counter[dataset_id])
                class_subsets.append(rng.choice(10, size=nb_classes, replace=False))

                if split == "train":
                    proportions.append(proportions_per_class[dataset_id])
                elif split == "val":
                    proportions.append(proportions_per_class[dataset_id] // 2)
                else:
                    proportions.append(None)

            super().__init__(
                datasets=datasets,
                proportions=proportions,
                class_counter=class_counter,
                class_subsets=class_subsets,
                seed=seed,
                split=split,
                target_size=(32, 32)
            )


But you can do your own stream by choosing the rules.
