CIFAR2Spurious
-----------------

CIFAR2Spurious is a scenario proposed in
`"Continual Feature Selection: Spurious Features in Continual" <https://arxiv.org/abs/2203.01012>`__.
This scenario is designed to highlights how continual learning algorithms rely on spurious features.
It is build from CIFAR10 and classes are reassigned between 0 and 1 in a balanced way (transportation mean vs not a transportation mean).
We can control the support and the correlation between the spurious feature and the label.


.. code-block:: python

    from continuum.datasets import CIFAR10
    from continuum.scenarios import CIFAR2Spurious

    #instanciate the dataset
    dataset_tr = CIFAR10("/your/path", train=True)
    dataset_te = CIFAR10("/your/path", train=False)

    # 5 tasks with same 2 classes
    scenario_tr = CIFAR2Spurious(dataset_tr, nb_tasks=5, seed=0, correlation=1.0, support=1.0,train=True)


    # / ! \ test scenario has nb_tasks+1 task, the last task contain all data without spurious feature (whatever the support).
    dataset_te = CIFAR2Spurious(dataset, nb_tasks=5, seed=0, correlation=1.0, support=1.0,train=False)

    # IMPORTANT
    # thanks to the seed, spurious features of scenario_tr and scenario_te will be the sames
    # so we can test performance on datasets with spurious features with scenario_te.
    # the difference of test with and without indicates the level of overfitting on spurious features.




