import os
from sqlite3 import DatabaseError

import numpy as np

import pytest

from continuum.datasets import MetaShift
from continuum.scenarios import ContinualScenario

DATA_PATH = os.environ.get("CONTINUUM_DATA_PATH")  

# NOTE : DATA_PATH folder should contain an "images" subfolder (not important if empty). 
#        Otherwise, it will try to download the full 20GB dataset.

@pytest.mark.slow
def test_metashift_with_class_names():

    dataset = MetaShift(DATA_PATH, download=True, train=True, class_names=["cat", "dog"])

    scenario = ContinualScenario(dataset)

    assert scenario.nb_classes == 2
    

@pytest.mark.slow
def test_metashift_with_nb_tasks():
   
    dataset = MetaShift(DATA_PATH, download=True, train=True, 
                        nb_tasks = 13)
    scenario = ContinualScenario(dataset)
    
    assert scenario.nb_tasks == 13
    

@pytest.mark.slow
def test_metashift_with_random_contexts():
    
    dataset = MetaShift(DATA_PATH, download=True, train=True, 
                        random_contexts=True, 
                        train_image_ids=["2317182", "2324913", "2383885"])
    
    x, y, t = dataset.get_data()

    assert len(x) == 3
    

