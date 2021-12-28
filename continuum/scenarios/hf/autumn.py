from typing import List

from continuum.scenarios.hf import HuggingFaceFellowship


class AutumnClassification(HuggingFaceFellowship):
    """A succession of HuggingFace datasets for text classification.

    Note that as specified in the original paper, ground-truth for amazon and yelp
    are the same, as they correspond to the same semantic targets (e.g. rating).

    All ground-truth are stored in the 'label' field. The input field are left
    unchanged and thus may differ depending on the dataset (some datasets have
    multiple input fields).

    * Episodic Memory in Lifelong Language Learning
      Cyprien de Masson d'Autume, Sebastian Ruder, Lingpeng Kong, Dani Yogatama
      NeurIPS 2019

    :param train: Train split vs test split.
    :param dataset_order: Order on which to load datasets. Default is
                          ['agnews', 'yelp', 'dbpedia', 'amazon', 'yahoo'].
    :param balanced: Re-sample randomly datasets as specified in the original paper.
                     115,000 examples per dataset for train, and 7,600 for test.
    :param seed: Random seed for the sampled if using balanced.
    """
    def __init__(
        self,
        train: bool = True,
        dataset_order: List[str] = ['agnews', 'yelp', 'dbpedia', 'amazon', 'yahoo'],
        balanced: bool = True,
        seed: int = 1
    ):
        self.dataset_order = dataset_order
        self.balanced = balanced
        self.seed = seed

        self.name_to_id = {
            "yelp": 'yelp_review_full',
            "amazon": ("amazon_reviews_multi", "en"),
            "agnews": 'ag_news',
            "dbpedia": 'dbpedia_14',
            "yahoo": 'yahoo_answers_topics',
        }
        self.name_to_cls = {
            "yelp": 5,
            "amazon": 5,
            "agnews": 4,
            "dbpedia": 14,
            "yahoo": 10
        }
        self.merged_classes = ["yelp", "amazon"]
        self.first_common_dataset = min(dataset_order.index(d) for d in self.merged_classes)
        for index, dataset in enumerate(self.dataset_order):
            if dataset not in self.merged_classes:
                continue
            if index == self.first_common_dataset:
                continue
            self.name_to_cls[dataset] = 0

        super().__init__(
            [self.name_to_id[name] for name in dataset_order],
            lazy=True,
            train=train
        )

    def __getitem__(self, index):
        dataset = super().__getitem__(index)

        if self.dataset_order[index] in self.merged_classes:
            index = self.first_common_dataset

        class_counter = sum(
            self.name_to_cls[name] for name in self.dataset_order[:index]
        )

        if 'topic' in dataset.column_names:  # for yahoo
            dataset = dataset.rename_column("topic", "label")
        if 'stars' in dataset.column_names:  # for amazon
            dataset = dataset.rename_column("stars", "label")
            class_counter -= 1

        def _closure(row):
            row['label'] += class_counter
            return row

        dataset = dataset.map(_closure)
        if self.balanced:
            if self.split == "train": size = 115000
            else: size = 7600

            dataset = dataset.train_test_split(test_size=size, seed=self.seed, shuffle=True)["test"]

        return dataset


class AutumnQA(HuggingFaceFellowship):
    pass  # TODO
