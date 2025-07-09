.PHONY: train
train:
	python lunarlander.py

.PHONY: eval
eval:
	python eval.py

.PHONY: collect_dataset
collect_dataset:
	python collect_dataset.py

.PHONY: baseline
bc:
	python pipelines/behavioral_cloning.py

.PHONY: clean
clean:
	rm -rf models/*
	rm -rf videos/*
	rm -rf runs/*
	rm -rf wandb/*
