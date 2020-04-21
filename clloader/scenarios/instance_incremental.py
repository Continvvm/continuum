

from clloader import CLLoader
from numpy as np


class InstanceIncremental(CLLoader):
    """Continual Loader, generating datasets for the consecutive tasks.
    Scenario: Classes are always the same but instances change (NI scenario)

    :param cl_dataset: A continual dataset.
    :param increment: Either number of classes per task, or a list specifying for
                      every task the amount of new classes.
    :param initial_increment: A different task size applied only for the first task.
                              Desactivated if `increment` is a list.
    :param train_transformations: A list of data augmentation applied to the train set.
    :param common_transformations: A list of transformations applied to both the
                                   the train set and the test set. i.e. normalization,
                                   resizing, etc.
    :param evaluate_on: How to evaluate on val/test, either on all `seen` classes,
                        on the `current` classes, or on `all` classes.
    :param class_order: An optional custom class order, used for NC.
    """
    # TODO


    # Vanilla NI: data are given randomly, each task has all classes and each task has different instances
    def _set_task_labels(self, y, increments):
        return np.random.randint(self.nb_tasks, size=len(y))