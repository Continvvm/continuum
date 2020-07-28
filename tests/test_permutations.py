import numpy as np
import pytest
from continuum.scenarios import Permutations
from tests.test_classorder import InMemoryDatasetTest
import torch
from torchvision.transforms import transforms


@pytest.fixture
def numpy_data():
    nb_classes = 6
    nb_data = 100

    x_train = []
    y_train = []
    x_train.append(np.array([np.random.randint(100, size=(2, 2, 3)).astype(dtype=np.uint8)] * nb_data))
    y_train.append(np.random.randint(nb_classes, size=(nb_data)))
    x_train = np.concatenate(x_train)
    y_train = np.concatenate(y_train)

    x_test = np.copy(x_train)
    y_test = np.copy(y_train)

    return (x_train, y_train.astype(int)), (x_test, y_test.astype(int))


'''
Test the initialization with three tasks
'''


@pytest.mark.parametrize("seed", [0, 42, 1664])
def test_init(numpy_data, seed):
    train, test = numpy_data
    dummy = InMemoryDatasetTest(*train)
    n_tasks = 3

    clloader = Permutations(cl_dataset=dummy, nb_tasks=n_tasks, seed=seed)

    # we recreate the list of seeds that Permutations class should have done
    g_cpu = torch.Generator()
    g_cpu.manual_seed(seed)
    list_seed = torch.randperm(1000, generator=g_cpu)[:n_tasks]
    list_seed[0] = 0  # first seed is always 0

    raw_samples = None
    ref_data = None

    for task_id, train_dataset in enumerate(clloader):
        assert task_id < n_tasks

        samples, _, _ = train_dataset.rand_samples(10)
        raw_samples, _, _ = train_dataset.get_raw_samples_from_ind(range(10))

        if task_id == 0:
            ref_data = samples
            raw_ref_data = raw_samples
        else:
            assert not torch.all(ref_data.eq(samples))

            # we verify that raw data are the same for all tasks
            assert (raw_ref_data == raw_samples).all()

            # we apply permutation manually
            x = transforms.ToTensor()(raw_samples[0])

            g_cpu.manual_seed(list_seed[task_id].item())
            shape = list(x.shape)

            x = x.reshape(-1)
            perm = torch.randperm(x.numel(), generator=g_cpu).long()
            x = x[perm]
            x = x.reshape(shape)

            # we compare manual permutation and permutation done by the class
            assert torch.all(samples[0].eq(x))
