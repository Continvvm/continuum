Useful functions for scenarios
--------------------

- ``create_subscenario(base_scenario, task_indexes)``

This function makes it possible to slice a scenario by selecting only a subset of tasks or
to reorder classes with a new order of task indexes.

.. code-block:: python
    from continuum.scenarios import create_subscenario
    # let say you have a continuum scenario with 5 tasks.
    scenario = some_func(nb_task=5)

    # this create a new scenario with only task 0 and 1 from the base_scenario
    subscenario = create_subscenario(base_scenario=scenario, task_indexes=[0,1])


    # this create a new scenario with a different order of task
    # Here the order of tasks is reversed.
    subscenario = create_subscenario(base_scenario=scenario, task_indexes=[4,3,2,1,0])


- ``get_scenario_remapping(scenario)->``
This function provided a remapping of class that ensure that labels comes in a continuous increasing order.
It is particularly useful if order of tasks has been changed (for example with ``create_subscenario``).
This function is often use with the function ``remap_class_vector`` which will apply the remapping.

.. code-block:: python
    from continuum.scenarios import create_subscenario, get_scenario_remapping, get_original_targets
    # let say you have a continuum scenario with 5 tasks.
    from continuum.datasets import MNIST

    scenario = ClassIncremental(
        MNIST(data_path="my/data/path", download=True, train=True),
        increment=2
     )

    # this create a new scenario with a different order of task
    # Here the order of tasks is reversed.
    subscenario = create_subscenario(base_scenario=scenario, task_indexes=[4,3,2,1,0])

    # here the class order of subscenario will be:
    # [task 0: (8,9)] -> [task 1: (6,7)] -> ...

    # get the remapping for the whole classes of the scenario
    remapping = get_scenario_remapping(subscenario)

    for taskset  in subscenario:
        loader = DataLoader(taskset)
        for x, y, t in loader:
            # remap the class vector here
            # (the made the process explicit on purpose in order to see where the classes are remapped)
            remap_y = remap_class_vector(y, mapping=remapping)

            #inverse remapping can be achieved with
            original_y = get_original_targets(remap_y, mapping=remapping)

            assert np.all(y==original_y) # should be true

The mapping can also be build online with `update_remapping(class_vector, mapping)` (it is more rigorous but it is also more computationnaly costly
and it does not change the final results).

.. code-block:: python
    from continuum.scenarios import create_subscenario, update_remapping, remap_class_vector

    scenario = some_scenario()

    # this create a new scenario with a different order of task
    # Here the order of tasks is reversed.
    subscenario = create_subscenario(base_scenario=scenario, task_indexes=[4,3,2,1,0])
    remapping = None

    for taskset  in subscenario:
        loader = DataLoader(taskset)
        for x, y, t in loader:
            # update online and automatically the mapping while receiving class vector
            remapping = update_remapping(y, remapping)
            # apply remapping after
            remap_y = remap_class_vector(y, mapping=remapping)

            #inverse remapping can be achieved with
            original_y = get_original_targets(remap_y, mapping=remapping)

            assert np.all(y==original_y) # should be true


- ``encode_scenario(scenario, model, batch_size, file_name, inference_fct=None)``

This function makes it possible to create a scenario with latent representation of a given model.
For example, when you have a frozen pretrained model and you want to just train the last layers.
With encode_scenario function, you can create a scenario with the data already encoded.
This function will save all the latent vectors into a hdf5 files and create the exact same initial scenario with encoded vectors.
It reduces the computation footprint and the time spent on encoding data for every experiences.

.. code-block:: python
    from continuum.scenarios import encode_scenario
    # let say you have a continuum scenario with 5 tasks.
    scenario = some_func(nb_task=5)
    feature_extractor = some_model()

    # inference function is an optional parameter to give function that will extract the latent representation you want.
    # by default
    inference_fct = (lambda model, x: model.to(torch.device('cuda:0'))(x.to(torch.device('cuda:0'))))

    # encode the scenario
    encoded_scenario = encode_scenario(scenario,
                                         feature_extractor,
                                         batch_size=64,
                                         file_name="encoded_scenario.hdf5",
                                         inference_fct=inference_fct)


    # the encoded_scenario can now be used like the original one but with encoded vector instead of original vectors.