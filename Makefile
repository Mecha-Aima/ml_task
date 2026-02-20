all: preprocess train evaluate

setup:
	mkdir -p data/processed models results
	python src/download_data.py

preprocess:
	python src/preprocess.py

train:
	python src/train.py

evaluate:
	python src/evaluate.py
