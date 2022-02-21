Continuum Scenarios
--------------------

Continual learning (CL) aim is to learn without forgetting in the most "satisfying" way. To evaluate the capacity of different CL algorithms the community use different type of benchmarks scenarios.

In Continuum, we implemented the four main types of scenarios used in the community: Class Incremental, Instance Incremental, New Class and Instance Incremental, and, Transformed Incremental. Those scenarios are originally suited for classification but they might be adapted for unsupervised learning, self-supervised learning or reinforcement learning.

For each scenario, there is a finite set of tasks and the goal is to learn the tasks one by one and to be able to generalize at the end to a test set composed of data from all tasks.

For clear evaluation purpose, the data from each tasks can not be found in another tasks. The idea is to be able to assess precisely what the CL algorithms is able to remember or not. Those benchmarks purpose is not to mimic real-life scenarios but to evaluate properly algorithms capabilities.

Here is roughly how continual learning scenarios are created and used:


.. code-block:: python

	from torch.utils.data import DataLoader

    # First we get a dataset that will be used to compose tasks and the continuum
    continual_dataset = MyContinualDataset()

    # Then the dataset is provided to a scenario class that will process it to create the sequence of tasks
    # the continuum might get specific argument, such as number of tasks.
    scenario = MyContinuumScenario(
        continual_dataset, SomeOtherArguments
    )

    # The continuum can then be enumerate tasks
    for task_id, taskset in enumerate(scenario):
          # taskset can be used as a Pytorch Dataset to load the task data
          loader = DataLoader(taskset)

          for x, y, t in loader:
                # data, label, task index
                # train on the task here
                break


A practical example with split MNIST:

.. code-block:: python

    from torch.utils.data import DataLoader

    from continuum import ClassIncremental
    from continuum.datasets import MNIST
    from continuum.tasks import split_train_val


    dataset = MNIST(data_path="my/data/path", download=True, train=True)

    # split MNIST with 2 classes per tasks -> 5 tasks
    scenario = ClassIncremental(dataset, increment=2)

    # The continuum can then be enumerate tasks
    for task_id, taskset in enumerate(scenario):

        # We use here a cool function to split the dataset into train/val with 90% train
        train_taskset, val_taskset = split_train_val(taskset, val_split = 0.1)
        train_loader = DataLoader(train_taskset)

         # train dataset is a normal data loader like in pytorch that can be used to load the task data
        for x, y, t in train_loader:
            # data, label, task index
            # train on the task here
            break

    # For testing we need to create another loader (It is importan to keep test a train separate)
    dataset_test = MNIST(data_path="my/data/path", download=True, train=False)


    # You can also create a test scenario to frame test data as train data.
    scenario_test = ClassIncremental(dataset_test, increment=2)

    # then iterate through tasks
    for task_id, test_taskset in enumerate(scenario_test):
        test_loader = DataLoader(test_taskset)
        for x, y, t in test_loader:
            # something
            break

    # you can also select specific task(s) in the continuum
    # It's just python slicing!
    # select task i
    i = 2
    test_taskset = continuum_test[i]

    # select tasks i to i+2
    test_taskset = continuum_test[i:i+2]

    # select all seen tasks up to the i-th task
    test_taskset = continuum_test[:i + 1]

    # select all tasks
    test_taskset = continuum_test[:]


Classes Incremental
--------------------

**In short:**
Each new task bring instances from new classes only.

**Aim:**
Evaluate the capability of an algorithms to learn concept sequentially, i.e. create representaion able to distinguish concepts and find the right decision boundaries without access to all past data.

**Some Details:**
The continuum of data is composed of several tasks. Each task contains class(es) that is/are specific to this task. One class can not be in several tasks.

One example, MNIST class incremental with five balanced tasks, MNIST has 10 classes then:
- task 0 contains data points labelled as 0 and 1
- task 1 contains data points labelled as 2 and 3
...
- task 4 contains data points labelled as 8 and 9

The Continual Loader *ClassIncremental* loads the data and batch it in several
tasks, each with new classes. See there some example arguments:

.. code-block:: python

    from torchvision.transforms import transforms
    from continuum.datasets import MNIST
    from continuum import ClassIncremental

    continual_dataset = MNIST(data_path="my/data/path", download=True, train=True)

    # first use case
    # first 2 classes per tasks
    scenario = ClassIncremental(
        continual_dataset,
        increment=2,
        transformations=[transforms.ToTensor()]
    )

    # second use case
    # first task with 2 classes then 4 classes per tasks until the end
    scenario = ClassIncremental(
        continual_dataset,
        increment=4,
        initial_increment=2,
        transformations=[transforms.ToTensor()]
    )

    # third use case
    # first task with 2, second task 3, third 1, ...
    scenario = ClassIncremental(
        continual_dataset,
        increment=[2, 3, 1, 4],
        transformations=[transforms.ToTensor()]
    )


A very important setting of Class Incremental scenarios is the class ordering. learning
'dog', then 'cat' may results in vastly different results than learning 'cat' then 'dog'.
It's very simple to change the class ordering:

.. code-block:: python

    from continuum import ClassIncremental

    continual_dataset = MNIST(data_path="my/data/path", download=True, train=True)

    # Default class ordering
    scenario_1 = ClassIncremental(
        continual_dataset,
        increment=2,
        class_order=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    )

    # Another class ordering
    scenario_2 = ClassIncremental(
        continual_dataset,
        increment=2,
        class_order=[1, 8, 3, 6, 5, 4, 10, 2, 9, 7]
    )


Note that as for every scenario in continuum, you have to create one scenario for the
training set, and one scenario for the testing set. Thus remember to use the same class
ordering for both!


Instance Incremental
--------------------

**In short:**
Each new tasks bring new instances from known classes.

**Aim:**
Evaluate the capability of an algorithms to improve its generalization capabilities through new data points, i.e. improve representation without access to all past data.

**Some Details:**
Tasks are made of new instances. By default the samples images are randomly
shuffled in different tasks, but some datasets provide, in addition of the data ``x`` and labels ``y``,
a task id ``t`` per sample. For example ``MultiNLI``, a NLP dataset, has 5 classes but
with 10 different domains. Each domain represents a new task.


.. code-block:: python

    from continuum import InstanceIncremental
    from continuum.datasets import MultiNLI

    dataset = MultiNLI("/my/path/where/to/download")
    scenario = InstanceIncremental(dataset=dataset)


Likewise, CORe50 provides domain ids which are automatically picked up by the `InstanceIncremental` scenario:


.. code-block:: python

    from continuum import InstanceIncremental
    from continuum.datasets import Core50v2_79, Core50v2_196, Core50v2_391

    scenario_79 = InstanceIncremental(dataset=Core50v2_79("/my/path"))
    scenario_196 = InstanceIncremental(dataset=Core50v2_196("/my/path"))
    scenario_391 = InstanceIncremental(dataset=Core50v2_391("/my/path"))


The three available version of CORe50 have respectively 79, 196, and 391 tasks. Each task may bring
new classes AND new instances of past classes, akin to the `NIC scenario <http://proceedings.mlr.press/v78/lomonaco17a.html>`_.


Another example could be using different dataset with their original classes as for MNISTFellowship:


.. code-block:: python

    from continuum import InstanceIncremental
    from continuum.datasets import MNISTFellowship

    # We create MNISTFellowship dataset and we keep original labels
    dataset = MNISTFellowship("/my/path/where/to/download", update_labels=False)
    scenario = InstanceIncremental(dataset=dataset, nb_tasks=3)


In this case, the three dataset MNIST, Fashion-MNIST and KMNIST will be learn with their original labels, e.g. classes 0 of all dataset stay 0.
Or with some other dataset:


.. code-block:: python

    from continuum import InstanceIncremental
    from continuum.datasets import CIFAR100

    dataset = CIFAR100("/my/path/where/to/download")
    scenario = InstanceIncremental(dataset=dataset, nb_tasks=42)

As you can see, for the last two examples, you need to provide the number of tasks. Because while MultiNLI
and CORe50 provide the domain/task ids of each sample, we don't have this information for other datasets
such as MNISTFellowhsip or CIFAR100. In this latter case, you must specify a number of tasks, and
then the dataset will be split randomly in this amount of tasks.


Instance incremental scenarios can also be create with transformation as described in next section.


Transformed Incremental
-----------------------

**In short:** Similar to instance incremental, each new tasks bring same instance with a different transformation (ex: images rotations, pixels permutations, ...)

**Aim:** Evaluate the capability of an algorithms to improve its generalization capabilities through new data points, i.e. improve representation without access to all past data.

**Some Details:**
The main difference with instance incremental, is that the scenarios builder has control of the different transformation spaces.
It is then easier to evaluate in which transformation space the algorithm is still able to generalize or not.

NB: the transformation used are `pytorch.transforms classes <https://pytorch.org/docs/stable/torchvision/transforms.html>`__

.. code-block:: python

    from continuum import TransformationIncremental

    list_of_transformation = [Trsf_0, Trsf_1, Trsf_2]

    # three tasks continuum, tasks 0 with Trsf_0 transformation
    scenario = TransformationIncremental(
        dataset=my_continual_dataset,
        incremental_transformations=list_of_transformation
    )




You can use TransformationIncremental along with the BackgroundSwap transform to create a domain incremental setting with changing backgrounds
This is inspired by the Mnist meta-sets from the following `paper <https://arxiv.org/abs/2007.02915>`__.

.. code-block:: python

    cifar = CIFAR10(DATA_PATH, train=True)
    mnist = MNIST(DATA_PATH, download=False, train=True)
    nb_task = 3
    list_trsf = []
    for i in range(nb_task):
        list_trsf.append([torchvision.transforms.ToTensor(), BackgroundSwap(cifar, bg_label=i, input_dim=(28, 28)),
                          torchvision.transforms.ToPILImage()])
    scenario = TransformationIncremental(mnist, base_transformations=[torchvision.transforms.ToTensor()],
                                         incremental_transformations=list_trsf)
    for task_id, task_data in enumerate(scenario):
        # Do magic here



- Permutations Incremental `source <https://github.com/Continvvm/continuum/blob/master/continuum/scenarios/permutations.py>`__
is a famous case of TransformationIncremental class, in this case the transformation is a fixed pixel permutation. Each task has a specific permutation.
The scenarios is then to learn a same task in various permutation spaces.

.. code-block:: python

    from continuum import Permutations
    from continuum.datasets import MNIST

    dataset = MNIST(data_path="my/data/path", download=True, train=True)
    nb_tasks = 5
    seed = 0

    # A sequence of permutations is initialized from seed `seed` each task is with different pixel permutation
    # shared_label_space=True means that all classes use the same label space
    # ex: an image of the zeros digit will be always be labelized as a 0 ( if shared_label_space=False, zeros digit image permutated will got another label than the original one)
    scenario = Permutations(cl_dataset=dataset, nb_tasks=nb_tasks, seed=seed, shared_label_space=True)

- Rotations Incremental `source <https://github.com/Continvvm/continuum/blob/master/continuum/scenarios/rotations.py>`__
is also a famous case of TransformationIncremental class, in this case the transformation is a rotation of image. Each task has a specific rotation or range of rotation.
The scenarios is then to learn a same task in various rotations spaces.

.. code-block:: python

    from continuum import Rotations
    from continuum.datasets import MNIST

    nb_tasks = 3
    # first example with 3 tasks with fixed rotations
    list_degrees = [0, 45, 90]
    # second example with 3 tasks with ranges of rotations
    list_degrees = [0, (40,50), (85,95)]

    dataset = MNIST(data_path="my/data/path", download=True, train=True)
    scenario = Rotations(
        cl_dataset=dataset,
        nb_tasks=nb_tasks,
        list_degrees=list_degrees
    )


Note that for all TransformationIncremental scenarios (included Rotations and Permutations) you can
use advanced indexing (e.g. `scenario[2:6]`, or `scenario[:7]`). In that case, when sampling multiple
tasks together, the same *original* images will be seen multiple times, but each time with the transformation
associated to the task.


New Class and Instance Incremental
----------------------------------

**In short:** Each new task bring both instances from new classes and new instances from known classes.

**Aim:** Evaluate the capability of an algorithms to both create new representation and improve existing ones.

**Some Details:**
NIC setting is a special case of NI setting. For now, only the CORe50 dataset
supports this setting.


The New class and instance incremental setting can be created with the Instance incremental class.
The `t` vector which define the task of each data point should be defined by the user or loaded from an existing scenario to create NIC scenario.

.. code-block:: python

    # given x data point, y class labels, t tasks labels
    # t define the tasks label for each data point.
    # Hence, the t vector define the number of tasks for the scenario and the order
    # task will be ordered in the croissant order

    from continuum import InstanceIncremental
    from continuum.datasets import InMemoryDataset
    NIC_Dataset = InMemoryDataset(x, y, t)
    NIC_Scenario = InstanceIncremental(NIC_Dataset)


Hashed Scenarios
----------------------------------

**In short:** Data ordering is determined by an hash function.

**Aim:** Evaluate the capability of an algorithms to learn new features independently from the labels.

**Some Details:**
This kind of scenario is proposed here to create tasks with related features in classical dataset and evaluate algorithm
capabilities in such context. This methodology force features to be different from one task to another
making the feature extractor training potentially harder. The resulting scenario (depending on the dataset)
is most probably a NIC scenario.


The New class and instance incremental setting can be created with the `HashedScenario` class.
The `t` vector is generate by the scenario class.
It can be saved and reloaded to avoid to compute the hash every time.

.. code-block:: python

    # given x data point, y class labels, t tasks labels
    # t define the tasks label for each data point.
    # Hence, the t vector define the number of tasks for the scenario and the order
    # task will be ordered in the croissant order

    from continuum import HashedScenario
    from continuum.datasets import CIFAR10

    dataset = CIFAR10(data_path="my/data/path", train=True)

    Hashed_Scenario = HashedScenario(dataset,
                                     nb_tasks=2,
                                     hash_name="AverageHash",
                                     transformations=None,
                                     filename_hash_indexes="hash_indexes_CIFAR10.npy",
                                     split_task = "balanced")


In this example we use "AverageHash" from `imagehash <https://github.com/JohannesBuchner/imagehash>`__ library but many other image hash can be used such as:
"AverageHash", "Phash", "PhashSimple", "DhashH", "DhashV", "Whash", "ColorHash" and "CropResistantHash".
More information about those hash function in imagehash `documentation <https://github.com/JohannesBuchner/imagehash>`__.

The `split_task` parameter can either be "balanced" or "auto", if the amount of data is balanced among tasks or
if it is set automatically to create more hash-coherent tasks.

If `split_task="auto"` and `nb_tasks=None` the number of tasks will be automatically estimated
with the `MeanShift` function from scikit-learn.

Incremental Semantic Segmentation
---------------------------------

Brought by `Michieli et al. ICCV-W 2017 <https://openaccess.thecvf.com/content_ICCVW_2019/papers/TASK-CV/Michieli_Incremental_Learning_Techniques_for_Semantic_Segmentation_ICCVW_2019_paper.pdf>`__
and `Cermelli et al. CVPR 2020 <https://arxiv.org/abs/2002.00718>`__, continual learning
for segmentation is very different from previous scenarios.

Semantic segmentation aims at classifying all pixels in an image, therefore
multiple classes can co-exist in the same image. This distinction leads to three kinds of scenarios:

- **Sequential**: where for a given task T, with current classes C, the model sees all
  images that contain at least one pixel labeled as a current classes C. If the image contains
  future classes, yet unseen, then it is discarded. In the sequential setting, all pixels
  are labeled, either with a old or current class label, background label (0), or
  unknown label (255).
- **Disjoint**: It's the same scenario as Sequential, but on one point. An image's pixel is
  only labeled for the current classes. Therefore, during training, if an old class is
  present in the image, its labels would be 0 (aka background). However, during the test
  phase, all labels (current + old) are present.
- **Overlap**: It's the same scenario as Disjoint except that the model can also
  see images containing a future class, as long as a current class is present.

Here is a quick example on how to do the challenging Overlap 15-1 scenario on Pascal-VOC2012:

.. code-block:: python

    from continuum.datasets import PascalVOC2012
    from continuum.scenarios import SegmentationClassIncremental
    from continuum.transforms.segmentation import ToTensor, Resize

    dataset = PascalVOC2012(
        data_path="/my/data/path/",
        train=True,
        download=True
    )

    scenario = SegmentationClassIncremental(
        dataset,
        nb_classes=20,
        initial_increment=15, increment=1,
        mode="overlap",
        transformations=[Resize((512, 512)), ToTensor()]
    )

NB: Following Cermelli et al., 15-1 means first a task of 15 classes, then followed by
multiple tasks made of new 1 class each.

Note that to build the different tasks, `SegmentationClassIncremental` has to
open every ground-truth segmentation maps which can take a few minutes. Therefore,
you can provide to the scenario the option `save_indexes="/path/where/to/save/indexes"`
that will save the computed task indexes. Then, if re-run a second time,
the scenario can quickly load the indexes.


Adding Your Own Scenarios with the `ContinualScenario` Class
----------------------------------

Continuum is developed to be flexible and easily adapted to new settings.
Then you can create a new scenario by providing simply a new dataset framed in an existing scenario such as Classes Incremental, Instance Incremental ...
You can also create a new class to create your own scenario with your own rules !

You can add it in the scenarios folder in the continuum project and make a pull request!

Scenarios can be seen as a list of `tasks <https://continuum.readthedocs.io/en/latest/_tutorials/datasets/tasks.html>`__ , the main thing to define is to define the content of each task to create a meaningful scenario.

You can also create personal scenarios simply by creating your own task label vector `t` with the
`ContinualScenario` Class. This class is made to just convert a cl_dataset into a scenario without any other processing.


.. code-block:: python

    from continuum.datasets import InMemoryDataset
    from continuum.scenarios import ContinualScenario

    x, y, t = fancy_data_generation_process()

    # t should contains the task label for each data point in x.
    # t should respect : np.unique(t).sort() == np.arange(len(np.unique(t)))

    cl_dataset = InMemoryDataset(x, y, t)
    scenario = ContinualScenario(cl_dataset)
