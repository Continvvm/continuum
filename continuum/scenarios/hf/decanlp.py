from continuum.scenarios.hf import HuggingFaceFellowship


class DecaNLP(HuggingFaceFellowship):
    """A decathlon of different HuggingFace datasets with different tasks.

    * The Natural Language Decathlon: Multitask Learning as Question Answering
      Bryan McCann and Nitish Shirish Keskar and Caiming Xiong and Richard Socher
      arXiv 2018
    """
    def __init__(self, train: bool = True):
        dataset_names = [
            "squad", "iwslt2017", "cnn_dailymail",
            "multi_nli", "sst", "qa_srl", "qa_zre",
            "woz_dialogue", "wikisql", "mwsc"
        ]

        super().__init__(
            dataset_names,
            lazy=True,
            train=train
        )
