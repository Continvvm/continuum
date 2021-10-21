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