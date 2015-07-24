.PHONY: build run
build:
	python setup.py build

run:
	python sofia_pipeline.py SoFiA_default_input.txt
