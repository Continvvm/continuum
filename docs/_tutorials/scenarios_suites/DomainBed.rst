DomainBed
-----------------

DomainBed is a collection of datasets for domain generalization proposed in
`"In Search of Lost Domain Generalization" <https://arxiv.org/abs/2007.01434>`__.

Each dataset is made of several "domains" where the pixels distribution change.

In the original paper, the authors tried different permutations where some domains
where uniquely in the train or test sets. You can reproduce exactly this setting in
Continuum. Although in the following examples, we instead share the same domains for train&test,
where each domain is a new task that we cannot review once learned.

We also assume that all classes are shared across domains. A new task doesn't bring new
classes as it does in `"ClassIncremental" <https://continuum.readthedocs.io/en/latest/_tutorials/scenarios/scenarios.html#classes-incremental>`__
but rather bring new domains as in `"InstanceIncremental" <https://continuum.readthedocs.io/en/latest/_tutorials/scenarios/scenarios.html#classes-incremental>`__
and ContinualScenario.

ColoredMNIST (CMNIST)
----------------------

The dataset is made of two labels: 0 & 1.
All the original digits from 0 to 4 are now 0, and the leftover are now 1.

25% of the labels have been randomly flipped.

Each label (0 & 1) has an assigned color (red or green). But flip_color% of
the samples have their color flipped. If the model learns the spurrious correlation
of the color, then it'll get super bad.

.. code-block:: python

    from continuum.datasets import ColoredMNIST, Fellowship
    from continuum import ContinualScenario

    dataset = Fellowship([
        ColoredMNIST("/your/path", train=True, download=True, flip_color=0.1),
        ColoredMNIST("/your/path", train=True, download=True, flip_color=0.2),
        ColoredMNIST("/your/path", train=True, download=True, flip_color=0.9),
    ])

    scenario = ContinualScenario(dataset)


RotatedMNIST (RMNIST)
----------------------


VLCS
----------------------

PACS
----------------------

OfficeHome
----------------------

TerraIncognita
----------------------


DomainNet
----------------------


