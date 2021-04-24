
distribute: FORCE clean
	python3 setup.py sdist bdist_wheel
	python3 -m twine upload --repository-url https://upload.pypi.org/legacy/ dist/*


prospector:
	prospector --profile-path .prospector.yaml continuum/

tests: FORCE
	pytest tests/  -m 'not slow'

fulltests: FORCE
	pytest tests/

fulltestsarthur: FORCE
	CONTINUUM_DATA_PATH=/local/douillard/ pytest tests/

fullteststim: FORCE
	CONTINUUM_DATA_PATH=../Datasets pytest tests/

clean: FORCE
	rm -rf dist/
	rm -rf continuum.egg-info/
	rm -rf tests/Datasets

documentation: FORCE
	$(MAKE) -C docs/ html

FORCE: ;
