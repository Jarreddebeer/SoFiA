.PHONY: build run test

build:
	python setup.py build

run:
	python sofia_pipeline.py

test:
	py.test
