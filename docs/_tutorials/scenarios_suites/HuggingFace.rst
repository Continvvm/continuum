HuggingFace's NLP datasets
----------------------------


HuggingFace proposes the largest amount of NLP datasets. The full list
can be found `here <https://huggingface.co/docs/datasets/>__`.

Naturally, Continuum can work with those NLP datasets and convert them in
continual scenarios.

A Continuum scenario can be iterated to produce task's dataset. For image-based datasets, those
tasks are implemented as `Taskset` that can be loaded with pytorch with a
`DataLoader`. For text-based datasets, the task's dataset is instead directly a HuggingFace's dataset.

HuggingFace Continual
=========================

`HuggingFaceIncremental` can allow you to split a HuggingFace dataset according
a to field name. For example, in the next code block, I download the MultiNLI dataset
with HuggingFace. Then, I'm asking to split this dataset according to the
`genre` field, and I'm also asking to see only 1 genre per task (`increment=1`):

.. code-block:: python


    import datasets  # from HuggingFace, do pip install datasets
    from continuum.scenarios.hf import HuggingFaceContinual

    multi_nli = datasets.load_dataset("multi_nli", split="train")

    scenario = HuggingFaceContinual(multi_nli, split_field="genre", increment=1)
    print(len(scenario), scenario.nb_classes)

    for task_dataset in scenario:
        print(task_dataset)


Note that all `task_dataset`s are also HuggingFace's datasets. So any functions you
used to apply one those (filtering, tokenization, etc.) you can still do it.

See below an example of what the "fields" could be in a HuggingFace dataset:

.. code-block:: bash

    >>> multi_nli
    Dataset({
        features: ['promptID', 'pairID', 'premise', 'premise_binary_parse', 'premise_parse', 'hypothesis', 'hypothesis_binary_parse', 'hypothesis_parse', 'genre', 'label'],
        num_rows: 392702
    })


HuggingFace Fellowship
=======================

`HuggingFaceFellowship` allows to chain multiple HuggingFace datasets one
after the other. Look at the following example where three datasets have been chained:


.. code-block:: python

    from continuum.scenarios.hf import HuggingFaceFellowship

    scenario = HuggingFaceFellowship(
        ["squad", "iwslt2017", "cnn_dailymail"],
        lazy=True,
        train=True
    )

    for dataset in scenario:
        print(dataset)


Each task will be made of only one dataset. With `HuggingFaceFellowship`,
you can specify a list of HuggingFace datasets, or a list of HuggingFace datasets names.
The latter is only string, and those names can be found on the HuggingFace documentation page.

You can also ask to load the dataset only on the fly with `lazy=True` instead
of loading all at the initialization.


Note that we proposes two pre-made fellowships that have been used previously in
the litterature: `AutumnQA` and `DecaNLP`.
