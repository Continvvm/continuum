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
[Colab](https://colab.research.google.com/drive/1KNd2sJ9nG9h33hI4C3Ec7KDJmFCR5cdA?usp=sharing) where
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
