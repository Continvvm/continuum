Synbols
-----------------

Synbols is a tool for rapidly generating new datasets with a rich composition of latent features rendered in low resolution images. Synbols leverages the large amount of symbols available in the Unicode standard and the wide range of artistic font provided by the open font community.

Continuum supports most of the official datasets found in the Synbols repository: https://github.com/ElementAI/synbols-resources/tree/master/datasets/generated under the "dataset_name" argument, as shown next. You can also generate your own dataset (see https://github.com/elementai/synbols).

Here is an usage example for InstanceIncremental learning.

.. code-block:: python

    from continuum import InstanceIncremental
    from continuum.datasets import Synbols

    synbols = InstanceIncremental(dataset=Synbols("/my/path", task_type="char", dataset_name="default_n=100000_2020-Oct-19.h5py"))


We also support the domain incremental case. For example, for the "char" classification scenario, we could introduce a new font in each task:

.. code-block:: python

    from continuum import ClassIncremental
    from continuum.datasets import Synbols

    synbols = ClassIncremental(dataset=Synbols("/my/path", task_type="char", domain_incremental_task="font",
                                                    dataset_name="default_n=100000_2020-Oct-19.h5py"))


