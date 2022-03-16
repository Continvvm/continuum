ALMA
-----------------

ALMA (On Anytime Learning At Macroscale) is a new framework, where like (offline) Continual Learning, data arrive sequentially in large (mega)batches over time. See paper `https://arxiv.org/abs/2106.09563`.
Unlike CL however, we do not assume that there is a shift in the underlying distribution. Rather, the goal of ALMA is develop strategies that perform well troughout the learning experience (not just at the end), and that do so efficiently from a compute and memory perspective. ALMA explore a different line of questions arising in this setting, namely : 

1. How long should a model wait and aggregate data before training again ?
2. Should the model increase its capacity over time to account for the additional data ?
 

Continuum Scenarios
####

ALMA is a different framing of `InstanceIncremental` therefore we focus on this scenario.


- Instance Incremental
""""""""

Class and Instances Incremental scenarios are proposed in the scenario from the original paper (next section).

.. code-block:: python

    from continuum.scenarios import ALMA
    scenario = ALMA(your_dataset, nb_megabatches=50)


