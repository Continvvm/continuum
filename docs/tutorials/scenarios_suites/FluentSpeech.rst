FluentSpeech
-----------------

Audio datasets with multiple targets (class id, action, object, location) that
represents `short commands <https://fluent.ai/fluent-speech-commands-a-dataset-for-spoken-language-understanding-research/>`__.

You need to install the `Soundfile <https://pypi.org/project/SoundFile/>`__ library.

You also probably want to add a custom transform to uniformize the length of each
audio in order to batch it.

.. code-block:: python

    from continuum import ClassIncremental, ContinualScenario
    from continuum.datasets import FluentSpeech
    from torch.utils.data import DataLoader

    dataset = FluentSpeech("/my/data/folder", train=True)

    def trunc(x, max_len):  # transformationn
        l = len(x)
        if l > max_len:
            x = x[l//2-max_len//2:l//2+max_len//2]
        if l < max_len:
            x = F.pad(x, (0, max_len-l), value=0.)
        return x

    # Iterates through the 31 possible classes
    scenario = ClassIncremental(
        dataset, increment=1, transformations=[partial(trunc, max_len=32000)])

    for taskset in scenario:
        loader = DataLoader(taskset, batch_size=32)

        for x, y, t in loader:
            print(x.shape, y.shape, t.shape, np.unique(y[:, 0]))
            break

    # Iterates through the 77 existing speakers
    scenario = ContinualScenario(dataset, transformations=[partial(trunc, max_len=32000)])

    for taskset in scenario:
        loader = DataLoader(taskset, batch_size=32)

        for x, y, t in loader:
            print(x.shape, y.shape, t.shape)
            break
