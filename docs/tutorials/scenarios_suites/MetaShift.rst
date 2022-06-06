MetaShift
---------

MetaShift can be described as a dataset of dataset. It divides the original Visual Genome dataset into classes and contexts in order to capture distribution shifts.
See paper : https://arxiv.org/abs/2202.06523

In continuum, we use the context information to build various types of scenarios, notably domain incremental scenario.

The user can select a subset of classes to include in the scenario with :code:`class_names` argument and if all class should appear in all tasks with :code:`strict_domain_inc` argument.

Images can be associated with several context, then the user can decide if one image appear only once or can appear several time.
If the image should appear only once, then the context chosen will be selected randomly.

Visual Genome:
##############

To download the Visual Genome dataset in advance :

.. code-block:: bash

    wget -c https://nlp.stanford.edu/data/gqa/images.zip
    unzip images.zip -d images


ContinualScenario:
##################

Beware that the default configuration will download the full Visual Genome dataset if it is not already present in a folder named "images" (20GB).

.. code-block:: python

    from continuum.datasets import MetaShift
    from continuum.scenarios import ContinualScenario

    data = MetaShift(datafolder)
    scenario = ContinualScenario(data)


- Specific Classes:

Select specific classes to appear in the dataset with the argument :code:`class_names`.
Then specify if all classes should apprear in all tasks with the argument :code:`strict_domain_inc`. If True, only contexts found in all classes will be kept.

.. code-block:: python

    from continuum.datasets import MetaShift
    from continuum.scenarios import ContinualScenario

    data = MetaShift(datafolder, class_names = ["cat", "dog"], strict_domain_inc = True)
    scenario = ContinualScenario(data)


- Specific image ids:

Select specific training iamage ids with the argument :code:`train_image_ids`.

.. code-block:: python

    from continuum.datasets import MetaShift
    from continuum.scenarios import ContinualScenario

    data = MetaShift(datafolder, train_image_ids=["2317182", "2324913", "2383885"])
    scenario = ContinualScenario(data)


- Get a unique task for each image:

Tasks will be choosen randomly for each image among all contexts corresponding to the image.

.. code-block:: python

    from continuum.datasets import MetaShift
    from continuum.scenarios import ContinualScenario

    data = MetaShift(datafolder, unique_occurence=True)
    scenario = ContinualScenario(data)
