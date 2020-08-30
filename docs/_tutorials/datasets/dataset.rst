Existing Datasets
-----------------

+----------------------+------------+------------+--------------------+--------+
|Name                  | Nb classes | Image Size | Automatic Download | Type   |
+======================+============+============+====================+========+
| **MNIST**            | 10         | 28x28x1    | YES                | Images |
+----------------------+------------+------------+--------------------+--------+
| **Fashion MNIST**    | 10         | 28x28x1    | YES                | Images |
+----------------------+------------+------------+--------------------+--------+
| **KMNIST**           | 10         | 28x28x1    | YES.               | Images |
+----------------------+------------+------------+--------------------+--------+
| **EMNIST**           | 10         | 28x28x1    | YES                | Images |
+----------------------+------------+------------+--------------------+--------+
| **QMNIST**           | 10         | 28x28x1    | YES                | Images |
+----------------------+------------+------------+--------------------+--------+
| **MNIST Fellowship** | 30         | 28x28x1    | YES                | Images |
+----------------------+------------+------------+--------------------+--------+
| **CIFAR10**          | 10         | 32x32x3    | YES                | Images |
+----------------------+------------+------------+--------------------+--------+
| **CIFAR100**         | 100        | 32x32x3    | YES                | Images |
+----------------------+------------+------------+--------------------+--------+
| **CIFAR Fellowship** | 110        | 32x32x3    | YES                | Images |
+----------------------+------------+------------+--------------------+--------+
| **ImageNet100**      | 100        | 224x224x3  | NO                 | Images |
+----------------------+------------+------------+--------------------+--------+
| **ImageNet1000**     | 1000       | 224x224x3  | NO                 | Images |
+----------------------+------------+------------+--------------------+--------+
| **CORe50**           | 50         | 224x224x3  | YES                | Images |
+----------------------+------------+------------+--------------------+--------+
| **CORe50-v2-79**     | 50         | 224x224x3  | YES                | Images |
+----------------------+------------+------------+--------------------+--------+
| **CORe50-v2-196***   | 50         | 224x224x3  | YES                | Images |
+----------------------+------------+------------+--------------------+--------+
| **CORe50-v2-391**    | 50         | 224x224x3  | YES                | Images |
+----------------------+------------+------------+--------------------+--------+
| **MultiNLI**         | 5          |            | YES                | Text   |
+----------------------+------------+------------+--------------------+--------+


All datasets have for arguments `train` and `download`, like a
`torchvision.dataset <https://pytorch.org/docs/stable/torchvision/datasets.html>`__. Those datasets are then modified to create `continuum scenarios <https://continuum.readthedocs.io/en/latest/_tutorials/scenarios/scenarios.html>`__.

Once a dataset is created, it is fed to a scenario that will split it in multiple tasks.

Continuum supports many datasets implemented in torchvision in such as **MNIST**, or **CIFAR100**:

.. code-block:: python

    from continuum import ClassIncremental
    from continuum.datasets import MNIST

    clloader = ClassIncremental(
        MNIST("/my/data/folder", download=True, train=True),
        increment=1,
        initial_increment=5
    )

The data from these small datasets can be automatically downloaded with the option `download`.

Larger datasets such as **ImageNet** or **CORe50** are also available, although their
initialization differ:

.. code-block:: python

    from continuum import ClassIncremental
    from continuum.datasets import ImageNet1000

    ImageNet1000("/my/data/folder/imagenet/train/", train=True)
    ImageNet1000("/my/data/folder/imagenet/val/", train=False)

Note that Continuum cannot download ImageNet's data, it's on you! We also provide ImageNet100,
a subset of 100 classes of ImageNet. The subset meta-data are automatically downloaded,
or you can provide them with the option `data_subset`.

Multiple versions of **CORe50** are proposed. For all, the data can automatically
be downloaded:

.. code-block:: python

    from continuum.datasets import Core50, Core50v2_196, Core50v2_391, Core50v2_79

    Core50("/data/data/folder/CORe50/", train=True, download=True)
    Core50v2_196("/data/douillard/CORe50/", train=True, download=True)
    Core50v2_391("/data/douillard/CORe50/", train=True, download=True)
    Core50v2_79("/data/douillard/CORe50/", train=True, download=True)

Refer to the datatset [official webpage](https://vlomonaco.github.io/core50/) for
more information about the different versions.

In addition to Computer Vision dataset, Continuum also provide one NLP dataset:

.. code-block:: python

    from continuum.datasets import MultiNLI

    MultiNLI("/my/data/folder", train=True, download=True)

The MultiNLI dataset provides text written in different styles and categories.
This dataset can be used in Continual Learning in a New Instances (NI) setting
where all categories are known from the start, but with styles being incrementally
added.

Adding Your Own Datasets
------------------------

The goal of continuum is to propose the most used benchmark scenarios of continual
learning but also to make easy the creation of new scenarios through an adaptable framework.

For example, the type of scenarios are easy to use with others dataset:

**InMemoryDataset**, for in-memory numpy array:

.. code-block:: python

    from continuum.datasets import InMemoryDataset

    x_train, y_train = gen_numpy_array()
    InMemoryDataset(x_train, y_train)


**PyTorchDataset**,for datasets defined in torchvision:

.. code-block:: python

    from torchvision.datasets import CIFAR10
    PyTorchDataset("/my/data/folder/", dataset_type=CIFAR10, train=True, download=True)


**ImageFolderDataset**, for datasets having a tree-like structure, with one folder per class:

.. code-block:: python

    from continuum.datasets import ImageFolderDataset

    ImageFolderDataset("/my/data/folder/train/")
    ImageFolderDataset("/my/data/folder/test/")

**Fellowship**, to combine several continual datasets.:

.. code-block:: python

    from torchvision.datasets import CIFAR10, CIFAR100
    from continuum.datasets import Fellowship

    Fellowship(data_path="/my/data/folder", dataset_list=[CIFAR10, CIFAR100])

Note that Continuum already provide pre-made Fellowship:

.. code-block:: python

    from continuum.datasets import MNISTFellowship, CIFARFellowship

    MNISTFellowship("/my/data/folder", train=True)
    CIFARFellowship("/my/data/folder", train=True)

You may want datasets that have a different transformation for each new task, e.g.
MNIST with different rotations or pixel permutations. Continuum also handles it!
However it's a scenario's speficic, not dataset, thus look over the
`Scenario doc <https://continuum.readthedocs.io/en/latest/_tutorials/scenarios/scenarios.html#transformed-incremental>`__.

