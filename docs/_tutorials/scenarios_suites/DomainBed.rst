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
##########

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
##########

MNIST with the same good old 10 digits. Although for each domains, the digits
are rotated by a certain amount.

.. code-block:: python

    from continuum.datasets import MNIST
    from continuum import Rotations

    dataset = MNIST("/your/path", train=True, download=True)

    scenario = Rotations(dataset, [15, 30, 45, 60])


VLCS
##########

A dataset of large images, with 5 classes (bird, car, chair, dog, and person)
distributed equally across 4 domains (Caltech101, LabelMe, SUN09, and VOC2007).


.. code-block:: python

    from torchvision import transforms
    from continuum.datasets import VLCS
    from continuum import ContinualScenario

    dataset = VLCS("/your/path", train=True, download=True)

    scenario = ContinualScenario(
        dataset,
        transformations=[transforms.Resize((224, 224)), transforms.ToTensor()]
    )


PACS
##########

A dataset of large images, with 7 classes distributed equally across 4 domains.
Note that you need to download yourself this dataset
`"here" <https://drive.google.com/file/d/0B6x7gtvErXgfbF9CSk53UkRxVzg/view>`__.


.. code-block:: python

    from torchvision import transforms
    from continuum.datasets import PACS
    from continuum import ContinualScenario

    dataset = PACS("/your/path", train=True, download=False)

    scenario = ContinualScenario(
        dataset,
        transformations=[transforms.Resize((224, 224)), transforms.ToTensor()]
    )


OfficeHome
##########

A dataset of large images, with 65 classes distributed equally across 4 domains.
Note that you need to download yourself this dataset
`"here" <https://drive.google.com/uc?id=0B81rNlvomiwed0V1YUxQdC1uOTg>`__.


.. code-block:: python

    from torchvision import transforms
    from continuum.datasets import OfficeHome
    from continuum import ContinualScenario

    dataset = OfficeHome("/your/path", train=True, download=False)

    scenario = ContinualScenario(
        dataset,
        transformations=[transforms.Resize((224, 224)), transforms.ToTensor()]
    )



TerraIncognita
##########


A dataset of large images, with 10 classes distributed equally across 4 domains.


.. code-block:: python

    from torchvision import transforms
    from continuum.datasets import TerraIncognita
    from continuum import ContinualScenario

    dataset = TerraIncognita("/your/path", train=True, download=False)

    scenario = ContinualScenario(
        dataset,
        transformations=[transforms.Resize((224, 224)), transforms.ToTensor()]
    )



DomainNet
##########

A dataset of large images, with 345 classes distributed equally across 6 domains.

.. code-block:: python

    from torchvision import transforms
    from continuum.datasets import DomainNet
    from continuum import ContinualScenario

    dataset = DomainNet("/your/path", train=True, download=False)

    scenario = ContinualScenario(
        dataset,
        transformations=[transforms.Resize((224, 224)), transforms.ToTensor()]
    )
