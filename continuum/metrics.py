import statistics
import collections
from functools import reduce


def require_subset(subset):
    def wrapper1(func):
        def wrapper2(self):
            if subset not in self._predictions:
                raise Exception(f"No {subset} predictions have been logged so far which last_accuracy rely on!")
            return func(self)
        return wrapper2
    return wrapper1


def cache(func):
    def wrapper(self):
        name = f"__cached_{func.__name__}"
        v = self.__dict__.get(name)
        if v is None:
            v = func(self)
            self.__dict__[name] = v
        return v
    return wrapper


class MetricsLogger:
    def __init__(self):
        self._predictions = collections.defaultdict(list)
        self._targets = collections.defaultdict(list)
        self._tasks = collections.defaultdict(list)
        self._model_sizes = []

    def add(self, predictions=None, targets=None, task_ids=None, subset="test", model=None):
        if subset not in ("train", "val", "test"):
            raise ValueError(f"Subset must be train, val, or test, not {subset}.")

        if predictions is not None and targets is not None and task_ids is not None:
            self._predictions[subset].append(predictions)
            self._targets[subset].append(targets)
            self._tasks[subset].append(task_ids)

        if model is not None:
            self._model_sizes.append(get_model_size(model))

        # Remove all cached properties
        for k in list(self.__dict__.keys()):
            if k.startswith("__cached_"):
                del self.__dict__[k]

    def log(self):
        print(f"Task id={self.nb_tasks}, acc={self.accuracy}, avg-acc={self.average_incremental_accuracy}")

    @property
    def nb_tasks(self):
        return len(self._predictions[list(self._predictions.keys())[0]])

    @property
    @cache
    @require_subset("test")
    def accuracy(self):
        return accuracy(
            self._predictions["test"][-1],
            self._targets["test"][-1]
        )

    @property
    @cache
    @require_subset("test")
    def accuracy_per_task(self):
        """Returns all task accuracy individually."""
        return [
            _get_R_ij(-1, j, all_preds, all_targets, all_tasks)
            for j in range(self.nb_tasks)
        ]

    @property
    @cache
    @require_subset("train")
    def online_cumulative_performance(self):
        """Computes the accuracy of last task on the train set.

        Reference:
        * Online Fast Adaptation and Knowledge Accumulation: a New Approach to Continual Learning
          Caccia et al. NeurIPS 2020
        """
        return accuracy(
            self._predictions["train"][-1],
            self._targets["train"][-1]
        )

    @property
    @cache
    @require_subset("test")
    def average_incremental_accuracy(self):
        """Computes the average of the accuracies computed after each task.

        Reference:
        * iCaRL: Incremental Classifier and Representation Learning
          Rebuffi et al. CVPR 2017
        """
        return statistics.mean([
            accuracy(self._predictions["test"][t], self._targets["test"][t])
            for t in range(len(self._predictions["test"]))
        ])

    @property
    @cache
    @require_subset("test")
    def backward_transfer(self):
        return backward_transfer(self._predictions["test"], self._targets["test"], self._tasks["test"])

    @property
    @cache
    @require_subset("test")
    def positive_backward_transfer(self):
        return positive_backward_transfer(self._predictions["test"], self._targets["test"], self._tasks["test"])

    @property
    @cache
    @require_subset("test")
    def remembering(self):
        return remembering(self._predictions["test"], self._targets["test"], self._tasks["test"])

    @property
    @cache
    @require_subset("test")
    def accuracy_A(self):
        return accuracy_A(self._predictions["test"], self._targets["test"], self._tasks["test"])

    @property
    @cache
    @require_subset("test")
    def forgetting(self):
        return forgetting(self._predictions["test"], self._targets["test"], self._tasks["test"])

    @property
    @cache
    def model_size_efficiency(self):
        return get_model_size_efficiency(self._model_sizes)


def accuracy(task_preds, task_targets):
    """Computes the accuracy of a given task.

    :param task_preds: Predicted labels.
    :param task_targets: Ground-truth targets.
    :return: a float metric between 0 and 1.
    """
    return (task_preds == task_targets).mean()


def accuracy_A(all_preds, all_targets, all_tasks):
    """Accuracy as defined in Diaz-Rodriguez and Lomonaco.

    Note that it is slightly different from the normal accuracy as it considers
    each task accuracy with equal weight, while the normal accuracy considers
    the proportion of all targets.

    Example:
    - Given task 1 with 50,000 images and task 2 with 1,000 images.
    - With normal accuracy, task 1 has more importance in the average accuracy.
    - With this accuracy A, task 1 has as much importance as task 2.

    Reference:
    * Don’t forget, there is more than forgetting: newmetrics for Continual Learning
      Diaz-Rodriguez and Lomonaco et al. NeurIPS Workshop 2018

    :param all_preds: All predicted labels up to now.
    :param all_targets: All targets up to now.
    :param all_tasks: All task ids up to now.
    :return: a float metric between 0 and 1.
    """
    T = len(all_preds)  # Number of seen tasks so far
    # TODO if we take in account zeroshot, we should take the max of all_tasks?
    A = 0.

    for i in range(T):
        for j in range(i+1):
            A += _get_R_ij(i, j, all_preds, all_targets, all_tasks)

    return A / (T * (T + 1) / 2)


def backward_transfer(all_preds, all_targets, all_tasks):
    """Measures the influence that learning a task has on the performance on previous tasks.

    Reference:
    * Gradient Episodic Memory for Continual Learning
      Lopez-paz & ranzato, NeurIPS 2017

    :param all_preds: All predicted labels up to now.
    :param all_targets: All targets up to now.
    :param all_tasks: All task ids up to now.
    :return: a float metric between 0 and 1.
    """
    T = len(all_preds)  # Number of seen tasks so far
    # TODO if we take in account zeroshot, we should take the max of all_tasks?
    if T <= 1:
        return 0.
    bwt = 0.

    for i in range(1, T):
        for j in range(0, i - 1):
            r_ij = _get_R_ij(i, j, all_preds, all_targets, all_tasks)
            r_jj = _get_R_ij(j, j, all_preds, all_targets, all_tasks)

            bwt += (r_ij - r_jj)

    return bwt / (T * (T - 1) / 2)


def positive_backward_transfer(all_preds, all_targets, all_tasks):
    """Computes the the positive gain of Backward transfer.

    Reference:
    * Don’t forget, there is more than forgetting: newmetrics for Continual Learning
      Diaz-Rodriguez and Lomonaco et al. NeurIPS Workshop 2018

    :param all_preds: All predicted labels up to now.
    :param all_targets: All targets up to now.
    :param all_tasks: All task ids up to now.
    :return: a float metric between 0 and 1.
    """
    bwt = backward_transfer(all_preds, all_targets, all_tasks)
    return 1 - abs(min(bwt, 0.))


def remembering(all_preds, all_targets, all_tasks):
    """Computes the forgetting part of Backward transfer.

    Reference:
    * Don’t forget, there is more than forgetting: newmetrics for Continual Learning
      Diaz-Rodriguez and Lomonaco et al. NeurIPS Workshop 2018

    :param all_preds: All predicted labels up to now.
    :param all_targets: All targets up to now.
    :param all_tasks: All task ids up to now.
    :return: a float metric between 0 and 1.
    """
    bwt = backward_transfer(all_preds, all_targets, all_tasks)
    return max(bwt, 0.)


def forward_transfer(all_preds, all_targets, all_tasks):
    """Measures the influence that learning a task has on the performance of future tasks.

    Reference:
    * Gradient Episodic Memory for Continual Learning
      Lopez-paz & ranzato, NeurIPS 2017

    :param all_preds: All predicted labels up to now.
    :param all_targets: All targets up to now.
    :param all_tasks: All task ids up to now.
    :return: a float metric between 0 and 1.
    """
    T = len(all_preds)  # Number of seen tasks so far
    # TODO if we take in account zeroshot, we should take the max of all_tasks?
    if T <= 1:
        return 0.

    fwt = 0.
    for i in range(T):
        for j in range(i):
            fwt += _get_R_ij(i, j)

    return bwt / (T * (T - 1) / 2)


def forgetting(all_preds, all_targets, all_tasks):
    """Measures the average forgetting.

    Reference:
    * Riemannian Walk for Incremental Learning: Understanding Forgetting and Intransigence
      Chaudhry et al. ECCV 2018
    """
    T = len(all_preds)  # Number of seen tasks so far
    # TODO if we take in account zeroshot, we should take the max of all_tasks?
    if T <= 1:
        return 0.

    f = 0.
    for j in range(T - 1):
        r_kj = _get_R_ij(T - 1, j, all_preds, all_targets, all_tasks)
        r_lj = max(_get_R_ij(l, j, all_preds, all_targets, all_tasks) for l in range(T - 1))
        f += (r_lj - r_kj)

    return f / (T - 1)


def _get_R_ij(i, j, all_preds, all_targets, all_tasks):
    """Computes an accuracy after task i on task j.

    R matrix:

          || T_e1 | T_e2 | T_e3
    ============================|
     T_r1 || R*  | R_ij  | R_ij |
    ----------------------------|
     T_r1 || R*  | R_ij  | R_ij |
    ----------------------------|
     T_r1 || R*  | R_ij  | R_ij |
    ============================|

    R_13 is the R of the first column and the third row.

    Reference:
    * Don’t forget, there is more than forgetting: newmetrics for Continual Learning
      Diaz-Rodriguez and Lomonaco et al. NeurIPS Workshop 2018

    :param i: Task id after which a model was trained.
    :param j: Task id of the test data.
    :param all_preds: All predicted labels up to now.
    :param all_targets: All targets up to now.
    :param all_tasks: All task ids up to now.
    :return: a float metric between 0 and 1.
    """
    preds = all_preds[i]
    targets = all_targets[i]
    tasks = all_tasks[i]

    indexes = tasks == j
    return (preds[indexes] == targets[indexes]).mean()


def get_model_size(model):
    """Computes the total number of parameters.

    :param model: A Pytorch's nn.Module model.
    :return: The number of parameters.
    """
    nb_params = 0

    for w in model.parameters():
        if len(w.shape) > 0:  # Tensor
            nb_params += reduce(lambda a, b: a * b, w.shape)
        else:  # Scalar
            nb_params += 1

    return nb_params


def get_model_size_efficiency(model_sizes):
    """Computes the efficiency of the model sizes.

    Reference:
    * Don’t forget, there is more than forgetting: newmetrics for Continual Learning
      Diaz-Rodriguez and Lomonaco et al. NeurIPS Workshop 2018

    :param model_sizes: A list of number of parameters, each being computed after a task.
    :return: a float metric between 0 and 1.
    """
    T = len(model_sizes)
    if T <= 1:
        return 1.0

    ms = 0.
    for i in range(T):
        ms += (model_sizes[0] / model_sizes[i])

    return min(1., ms / T)
