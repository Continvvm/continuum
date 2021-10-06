Rehearsal
---------

A popular and efficient method in continual learning is to do **rehearsal**. Aka
the action of reviewing a limited amount of previous images.

Continuum provides a helper class to manage this memory and functions to
samples ("to herd") examples into the memory.

How to use the memory:

.. code-block:: python

    from torch.utils.data import DataLoader

    from continuum import ClassIncremental
    from continuum.datasets import CIFAR100
    from continuum import rehearsal

    scenario = ClassIncremental(
        CIFAR100(data_path="my/data/path", download=True, train=True),
        increment=10,
        initial_increment=50
    )

    memory = rehearsal.RehearsalMemory(
        memory_size=2000,
        herding_method="barycenter"
    )

    for task_id, taskset in enumerate(scenario):
        if task_id > 0:
            mem_x, mem_y, mem_t = memory.get()
            taskset.add_samples(mem_x, mem_y, mem_t)

        loader = DataLoader(taskset, shuffle=True)
        for epoch in range(epochs):
            for x, y, t in loader:
                # Do your training here

        # Herding based on the barycenter (as iCaRL did) needs features,
        # so we need to extract those features, but beware to use a loader
        # without shuffling.
        loader = DataLoader(taskset, shuffle=False)

        features = my_function_to_extract_features(my_model, loader)

        memory.add(
            *taskset.get_raw_samples(), features
        )


Herding
-------

We predefine three methods to herd new samples:
- `random`: random (quite efficient despite its simplicity...)
- `cluster`: samples whose features are closest to their class mean feature
- `barycenter`: samples whose features are closest to a moving barycenter as iCaRL did

If you want to define your own herding method and provide it to `RehearsalMemory`
(instead of the string 'barycenter' as in the previous example), you should:

1. Take three parameters in arguments `x`, `y`, `t`, `z`, `nb_per_classes`.

`x`, `y`, `t` are the input data, targets and task ids. `z` is an extra info,
that can be whatever you want. For 'barycenter' and 'cluster' it's features.

2. Returns the sampled `x`, `y`, `t`. Note that you could even create new `x`, `y`, `t`
   and returns those if you want.


Saving and loading
-------------------

Computing rehearsal samples can be slow, and thus if you want to re-start from a
checkpoint, you would also want to avoid re-computing the herding. Thus, the memory
provide a save and load methods:

.. code-block:: python

    memory.save("/my/path/memory.npz")
    memory.load("/my/path/memory.npz")
