.PHONY: fm
fm:
	@echo "Running flow matching experiment..."
	./experiments/fm.sh

.PHONY: train
train:
	./experiments/lunarlander/train.sh

.PHONY: eval
eval:
	python eval.py

.PHONY: collect_dataset
collect_dataset:
	python pipelines/collect_dataset.py

.PHONY: baseline_bc
baseline_bc:
	python pipelines/baseline_bc.py

.PHONY: fm_bc
fm_bc:
	python fm_bc.py

.PHONY: clean
clean:
	rm -rf models/*
	rm -rf videos/*
	rm -rf runs/*
	rm -rf wandb/*
