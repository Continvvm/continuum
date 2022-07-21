import os
from sqlite3 import DatabaseError

import numpy as np

import pytest

from continuum.datasets import MetaShift
from continuum.scenarios import ContinualScenario

DATA_PATH = os.environ.get("CONTINUUM_DATA_PATH")
VISGEN_PATH = os.environ.get("VISGEN_PATH")
if VISGEN_PATH == "":
    VISGEN_PATH = None
# NOTE : DATA_PATH folder should contain an "images" subfolder (not important if empty). 
#        Otherwise, it will try to download the full 20GB dataset.

@pytest.mark.slow
def test_metashift_with_class_names():

    dataset = MetaShift(DATA_PATH, visual_genome_path=VISGEN_PATH, download=True, train=True, class_names=["cat", "dog", "horse"], strict_domain_inc=True)

    scenario = ContinualScenario(dataset)

    assert scenario.nb_classes == 3

@pytest.mark.slow
def test_metashift_with_nb_tasks():
   
    dataset = MetaShift(DATA_PATH, visual_genome_path=VISGEN_PATH, download=True, train=True, 
                        nb_tasks = 13)
    scenario = ContinualScenario(dataset)
    
    assert scenario.nb_tasks == 13
    
@pytest.mark.slow
def test_metashift_with_unique_occurence():
    
    dataset = MetaShift(DATA_PATH, visual_genome_path=VISGEN_PATH, download=True, 
                        train=True,
                        class_names = ['cat', 'dog'],
                        context_names = ['water', 'computer'],
                        unique_occurence=True)
    
    x,_,_ = dataset.get_data()
    dogs_n_cats, contexts = dataset.get_class_context_in_order()
    
    assert (np.array(['cat', 'dog']) == dogs_n_cats).all() or (np.array(['dog', 'cat']) == dogs_n_cats).all()
    assert (np.array(['water', 'computer']) == contexts).all() or (np.array(['computer', 'water']) == contexts).all()

    assert len(x) == len(np.unique(x))

