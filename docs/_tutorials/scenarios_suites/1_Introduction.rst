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
