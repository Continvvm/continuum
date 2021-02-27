## Introduction

In the continual learning litterature, several evaluation benchmarks suite have been proposed. Those suites propose a variety of tasks to evaluate algorithms. 

- CORe50
- Stream-51
- And soon many others



## CORe50

- Class Incremental



- Instance Incremental




- Class and Instance Incremental


CORe50 provides domain ids which are automatically picked up by the `InstanceIncremental` scenario:


.. code-block:: python

    from continuum import InstanceIncremental
    from continuum.datasets import Core50v2_79, Core50v2_196, Core50v2_391

    scenario_79 = InstanceIncremental(dataset=Core50v2_79("/my/path"))
    scenario_196 = InstanceIncremental(dataset=Core50v2_196("/my/path"))
    scenario_391 = InstanceIncremental(dataset=Core50v2_391("/my/path"))


The three available version of CORe50 have respectively 79, 196, and 391 tasks. Each task may bring
new classes AND new instances of past classes, akin to the `NIC scenario <http://proceedings.mlr.press/v78/lomonaco17a.html>`_.

## Synbols

Synbols is a tool for rapidly generating new datasets with a rich composition of latent features rendered in low resolution images. Synbols leverages the large amount of symbols available in the Unicode standard and the wide range of artistic font provided by the open font community. 

Continuum supports most of the official datasets found in the Synbols repository: https://github.com/ElementAI/synbols-resources/tree/master/datasets/generated under the "dataset_name" argument, as shown next. You can also generate your own dataset (see https://github.com/elementai/synbols).

Here is an usage example for InstanceIncremental learning.

.. code-block:: python

    from continuum import InstanceIncremental
    from continuum.datasets import Synbols
    
    synbols = InstanceIncremental(dataset=Synbols("/my/path", task_type="char", dataset_name="default_n=100000_2020-Oct-19.h5py"))
    

We also support the domain incremental case. For example, for the "char" classification scenario, we could introduce a new font in each task:

.. code-block:: python

    from continuum import TaskIncremental
    from continuum.datasets import Synbols
    
    synbols = TaskIncremental(dataset=Synbols("/my/path", task_type="char", domain_incremental_task="font",
                                                    dataset_name="default_n=100000_2020-Oct-19.h5py"))

