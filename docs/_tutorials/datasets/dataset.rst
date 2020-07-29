Existing Datasets
-----------------

Most of the dataset used here come from [torchvision.dataset](https://pytorch.org/docs/stable/torchvision/datasets.html). Those datasets are then modified to create [continuum scenarios](https://continuum.readthedocs.io/en/latest/_tutorials/scenarios/scenario.html).


Adding Your Own Datasets
------------------------

The goal of continuum is to propose the most used benchmark scenarios of continual learning but also to make easy the creation of new scenarios through an adaptable framework.

For example, the type of scenarios are easy to use with others dataset:

**InMemoryDataset**, for in-memory numpy array:
```python
x_train, y_train = gen_numpy_array()

clloader = CLLoader(
    InMemoryDataset(x_train, y_train),
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
