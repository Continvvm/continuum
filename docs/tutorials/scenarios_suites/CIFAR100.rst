CIFAR100
-----------------

CIFAR100 is a famous dataset proposed in
`"Learning Multiple Layers of Features from Tiny Images (pdf)" <https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf>`__.
This dataset is mainly used with its 100 class labels type. However, it exists also 20 super classes (coarse labels or category labels).
In continuum, we propose to benefit from both to create various types of scenarios.

Continuum Scenarios
##########

- Class Incremental
""""""""

We can create a simple class incremental setting with the default parameters, i.e. 100 classes.
In this scenario coarse labels are not used.

.. code-block:: python

    from continuum.datasets import CIFAR100
    from continuum.scenarios import ClassIncremental

    dataset = CIFAR100("/your/path", train=True)
    # 5 tasks with 20 classes each
    scenario = ClassIncremental(dataset, nb_tasks=5)


Or a `ClassIncremental` with coarse labels (category labels).
In this scenario classical labels are not used.

.. code-block:: python

    from continuum.datasets import CIFAR100
    from continuum.scenarios import ClassIncremental

    dataset = CIFAR100("/your/path", train=True, labels_type="category")
    # 5 tasks with 4 classes each
    scenario = ClassIncremental(dataset, nb_tasks=5)


- Instance Incremental (lifelong)
""""""""

We can create a instance incremental setting with the coarse labels, i.e. 20 classes.
Data are labeled with the coarse labels of CIFAR100.
However, data are shared between tasks using the original label to ensure a domain drift between tasks
, e.g., for the coarse label \say{aquatic mammals} the data go from beavers to dolphins to otters to seals to finally whales in separate tasks.

.. code-block:: python

    from continuum.datasets import CIFAR100
    from continuum.scenarios import ContinualScenario

    dataset = CIFAR100("/your/path",
                        train=True,
                        labels_type="category",
                        task_labels="lifelong")
    # 5 tasks with 20 labels (coarse labels) each (always the same labels but different classes)
    scenario = ContinualScenario(dataset)


- Classes and Instances Incremental
""""""""

In Class and Instances Incremental scenario, the labels are set by category of object but new tasks bring new object.
Hence, new task either bring a new object from a known category or a new object from an unknown category.

.. code-block:: python

    from continuum.datasets import CIFAR100
    from continuum.scenarios import ContinualScenario
    # task_labels parameter makes possible to create a task id vector from either classes or categories.
    dataset = CIFAR100("/your/path", train=True), task_labels="class", labels_type="category"
    # 100 tasks with 1 object each among the 20 categories of coarse labels
    # classes are object ids (20 classes then), new tasks might contains new label or known label
    scenario = ContinualScenario(dataset)

