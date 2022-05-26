MetaShift
-----------------

MetaShift can be described as a dataset of dataset. It divides the original Visual Genome dataset into classes and contexts in order to capture distribution shifts.
See paper : https://arxiv.org/abs/2202.06523

The MetaShift class is built with ContinualScenario in mind. But it also lets the user select specific classes or training ids. 
As many images can be found in more than one class(context) configuration, the user can specify that all images should aprear in exactly one configuration.

Visual Genome:
##############

To download the Visual Genome dataset in advance :

.. code-block:: bash

    wget -c https://nlp.stanford.edu/data/gqa/images.zip
    unzip images.zip -d images


Images will have to be stored this way :

.. code-block:: bash

    data_path/images/images/id.jpg


ContinualScenario:
##################

Beware that the default configuration will download the full Visual Genome dataset if it is not already present in a folder named "images".

.. code-block:: python

    from continuum.datasets import MetaShift
    from continuum.scenarios import ContinualScenario

    data = MetaShift(datafolder)
    scenario = ContinualScenario(data)


- Specific Classes:

.. code-block:: python

    from continuum.datasets import MetaShift
    from continuum.scenarios import ContinualScenario

    data = MetaShift(datafolder, class_names = ["cat", "dog"])
    scenario = ContinualScenario(data)


- Specific ids:

.. code-block:: python

    from continuum.datasets import MetaShift
    from continuum.scenarios import ContinualScenario

    data = MetaShift(datafolder, train_image_ids=["2317182", "2324913", "2383885"])
    scenario = ContinualScenario(data)


- Get a unique task for each image:

.. code-block:: python

    from continuum.datasets import MetaShift
    from continuum.scenarios import ContinualScenario

    data = MetaShift(datafolder, random_contexts=Ture)
    scenario = ContinualScenario(data)
