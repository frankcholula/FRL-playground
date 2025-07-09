.PHONY: train
train:
	python lunarlander.py

.PHONY: eval
eval:
	python eval.py

.PHONY: collect_dataset
collect_dataset:
	python collect_dataset.py

.PHONY: baseline_bc
baseline_bc:
	python pipelines/baseline_bc.py

.PHONY: clean
clean:
	rm -rf models/*
	rm -rf videos/*
	rm -rf runs/*
	rm -rf wandb/*
