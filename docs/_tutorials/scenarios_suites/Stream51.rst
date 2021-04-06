Stream 51
-----------------

(This part of the documentation is probably not up to date with https://github.com/Continvvm/continuum/pull/111)

Two instance incremental scenarios: based on clip or video


The clip dataset with InstanceIncremental scenario is equivalent to the "instance" scenario of the original Stream-51 paper.
.. code-block:: python

    from torchvision.transforms import Resize, ToTensor

    from continuum.datasets import Stream51
    from continuum.scenarios import InstanceIncremental


    dataset = Stream51('../Datasets', task_criterion="clip")
    scenario = InstanceIncremental(dataset, transformations=[Resize((224, 224)), ToTensor()])

    for task_id, task_set in enumerate(scenario):
        task_set.plot(path="Archives/Samples/Stream51",
                         title="Stream51_InstanceIncremental_{}.jpg".format(task_id),
                         nb_samples=100)

The video dataset with InstanceIncremental scenario is not present in the original Stream-51 paper.
It proposes to learn from sequence of videos without cutting them into clips.
.. code-block:: python

    from torchvision.transforms import Resize, ToTensor

    from continuum.datasets import Stream51
    from continuum.scenarios import InstanceIncremental


    dataset = Stream51('../Datasets', task_criterion="video")
    scenario = InstanceIncremental(dataset, transformations=[Resize((224, 224)), ToTensor()])

    for task_id, task_set in enumerate(scenario):
        task_set.plot(path="Archives/Samples/Stream51/video",
                         title="Stream51_InstanceIncremental_{}.jpg".format(task_id),
                         nb_samples=100)

The clip dataset with ClassIncremental scenario is equivalent to the "instance_class" scenario of the original Stream-51 paper.
.. code-block:: python

    from torchvision.transforms import Resize, ToTensor

    from continuum.datasets import Stream51
    from continuum.scenarios import ClassIncremental


    dataset = Stream51('../Datasets', task_criterion="video")
    scenario = ClassIncremental(dataset, transformations=[Resize((224, 224)), ToTensor()])

    for task_id, task_set in enumerate(scenario):
        task_set.plot(path="Archives/Samples/Stream51/video",
                         title="Stream51_InstanceIncremental_{}.jpg".format(task_id),
                         nb_samples=100)


iid scenario

.. code-block:: python

    from torchvision.transforms import Resize, ToTensor

    from continuum.datasets import Stream51
    from continuum.scenarios import InstanceIncremental


    dataset = Stream51('/path/to/data')
    scenario = InstanceIncremental(dataset, transformations=[Resize((224, 224)), ToTensor()])
    unique_task_set = scenario[:]