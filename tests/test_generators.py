import numpy as np
import pytest
import torch
import random
import string

from continuum.datasets import InMemoryDataset
from continuum.scenarios import ClassIncremental
from continuum.generators import TaskOrderGenerator, ClassOrderGenerator, HashGenerator


def gen_data():
    x_train = np.random.randint(0, 255, size=(20, 32, 32, 3))
    y_train = []
    for i in range(10):
        y_train.append(np.ones(2) * i)
    y_train = np.concatenate(y_train)

    x_test = np.random.randint(0, 255, size=(20, 32, 32, 3))
    y_test = np.copy(y_train)

    return (x_train, y_train), (x_test, y_test)


def test_task_order_generator():
    train, test = gen_data()
    dummy = InMemoryDataset(*train)
    scenario = ClassIncremental(dummy, increment=1)

    scenario_generator = TaskOrderGenerator(scenario)
    sample_scenario = scenario_generator.sample()

    assert sample_scenario.nb_tasks == scenario.nb_tasks


def test_class_order_generator():
    train, test = gen_data()
    dummy = InMemoryDataset(*train)
    scenario = ClassIncremental(dummy, increment=1)

    scenario_generator = ClassOrderGenerator(scenario)
    sample_scenario = scenario_generator.sample()

    assert sample_scenario.nb_tasks == scenario.nb_tasks
    assert sample_scenario.nb_classes == scenario.nb_classes
    assert (sample_scenario.classes == scenario.classes).all()


@pytest.mark.parametrize("seed",
                         [0, 41, 1992]
                         )
def test_class_order_generator(seed):
    train, test = gen_data()
    dummy = InMemoryDataset(*train)
    scenario = ClassIncremental(dummy, increment=1)

    scenario_generator = ClassOrderGenerator(scenario)
    sample_scenario = scenario_generator.sample(seed)
    class_order = scenario_generator.get_class_order(seed)

    assert (np.array(class_order) == np.array(sample_scenario.class_order)).all()


@pytest.mark.parametrize("seeds", [
    [0, 1],
    [1664, 41],
    [1792, 1992]
])
def test_task_order_generator_seed(seeds):
    train, test = gen_data()
    seed_0 = seeds[0]
    seed_1 = seeds[1]
    dummy = InMemoryDataset(*train)
    scenario = ClassIncremental(dummy, increment=1)

    scenario_generator = TaskOrderGenerator(scenario)
    task_order_0 = scenario_generator.get_task_order(seed=seed_0)
    task_order_1 = scenario_generator.get_task_order(seed=seed_1)
    task_order_0_2 = scenario_generator.get_task_order(seed=seed_0)

    assert not torch.all(task_order_0.eq(task_order_1))
    assert torch.all(task_order_0.eq(task_order_0_2))


@pytest.mark.parametrize("nb_tasks", [
    2,
    4
])
def test_task_order_generator_nb_tasks(nb_tasks):
    train, test = gen_data()
    dummy = InMemoryDataset(*train)
    scenario = ClassIncremental(dummy, increment=1)
    scenario_generator = TaskOrderGenerator(scenario)
    sample_scenario = scenario_generator.sample(nb_tasks=nb_tasks)

    assert sample_scenario.nb_tasks == nb_tasks


def test_hash_generator_auto():
    train, test = gen_data()
    dummy = InMemoryDataset(*train)
    NB_TASKS = 2
    list_hash = ["AverageHash", "Phash", "PhashSimple", "DhashH", "DhashV", "Whash", "ColorHash"
                 ] # , "CropResistantHash"
    scenario_generator = HashGenerator(cl_dataset=dummy,
                                       list_hash=list_hash,
                                       nb_tasks=2,
                                       transformations=None,
                                       filename_hash_indexes=None,
                                       split_task="auto")

    sample_scenario = scenario_generator.sample()

    assert sample_scenario.nb_tasks == NB_TASKS


def test_hash_generator_auto_full():
    train, test = gen_data()
    dummy = InMemoryDataset(*train)
    list_hash = ["AverageHash", "Phash", "PhashSimple", "DhashH", "DhashV", "Whash", "ColorHash"
                 ] # , "CropResistantHash"
    scenario_generator = HashGenerator(cl_dataset=dummy,
                                       list_hash=list_hash,
                                       nb_tasks=None,
                                       transformations=None,
                                       filename_hash_indexes=None,
                                       split_task="auto")

    sample_scenario = scenario_generator.sample()
    assert sample_scenario.nb_tasks >= 2
    sample_scenario = scenario_generator.sample()
    assert sample_scenario.nb_tasks >= 2


@pytest.mark.parametrize("nb_tasks, list_hash_name", [
    (2, ["AverageHash", "Whash", "ColorHash", "DhashV"]), #"CropResistantHash"
    (4, ["DhashH", "DhashV", "Whash", "ColorHash"]),
    (3, ["AverageHash", "DhashH"]),
    (5, ["AverageHash", "Phash", "PhashSimple", "DhashH", "DhashV", "Whash", "ColorHash",
         ]) #"CropResistantHash"
])
def test_hash_generator_balanced(nb_tasks, list_hash_name):
    train, test = gen_data()
    dummy = InMemoryDataset(*train)

    print(list_hash_name)
    scenario_generator = HashGenerator(cl_dataset=dummy,
                                       list_hash=list_hash_name,
                                       nb_tasks=nb_tasks,
                                       transformations=None,
                                       filename_hash_indexes=None,
                                       split_task="balanced")

    sample_scenario = scenario_generator.sample()
    assert sample_scenario.split_task == "balanced"
    assert sample_scenario.hash_name in list_hash_name
    # test nb of samples per task
    assert (len(sample_scenario[0]) == len(sample_scenario[1]))\
           or (len(sample_scenario[0]) + 1 == len(sample_scenario[1]))
