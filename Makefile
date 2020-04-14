


distribute: FORCE clean
	python3 setup.py sdist bdist_wheel
	python3 -m twine upload --repository-url https://upload.pypi.org/legacy/ dist/*


tests: FORCE
	pytest tests/

clean: FORCE
	rm -rf dist/
	rm -rf clloader.egg-info/

FORCE: ;
