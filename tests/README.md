

# Run Tests


Command to run from the main folder:

```bash
#python 3.6
# All test
python -m pytest tests/

# one test, ex
python -m pytest tests/test_Dataloader.py 

# stop after first failure
pytest -x          

# run a specific test within a module:
pytest test_mod.py::test_func

# run slow tests
python -m pytest -m slow

# get coverage
python -m pytest --cov=. tests/
```
