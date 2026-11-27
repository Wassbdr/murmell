PYTHON ?= python
BENCH_ARGS ?=
VARIANT ?= round1
SIM_ARGS ?=
INVISIBLE_WORK ?= work/round3/trustmark-bench
TRUSTMARK_STEM := $(INVISIBLE_WORK)/trustmark-decoder-128-0-1600

.DEFAULT_GOAL := help
.PHONY: help install test test-task1 lint bench visible-bench visible-audit visible-shipped visible-weights weights task2 task3 showcase

help:
	@printf '%s\n' 'install test lint bench weights visible-bench visible-audit visible-shipped visible-weights task2 task3 showcase'

install:
	$(PYTHON) -m pip install -r benchmarks/requirements.txt -e '.[app,onnx,dev,showcase]'
	$(PYTHON) -m playwright install chromium

test: test-task1
	cd task2_model_equality && $(PYTHON) -m pytest -q tests
	cd task3_unlearning && $(PYTHON) -m pytest -q tests

test-task1:
	PYTHONPATH=task1_watermark_localisation $(PYTHON) -m pytest -q task1_watermark_localisation/tests experiments/visible/tests

lint:
	$(PYTHON) -m ruff check .
	$(PYTHON) -m mypy

bench:
	cd task1_watermark_localisation && $(PYTHON) -m src.invisible_bench --work "$(INVISIBLE_WORK)" $(BENCH_ARGS)

visible-bench:
	PYTHONPATH=task1_watermark_localisation $(PYTHON) -m experiments.visible.src.bench $(BENCH_ARGS)

visible-audit:
	PYTHONPATH=task1_watermark_localisation $(PYTHON) -m experiments.visible.src.audit_watermarks

visible-shipped:
	PYTHONPATH=task1_watermark_localisation $(PYTHON) -m experiments.visible.src.shipped_report

visible-weights:
	@test '$(VARIANT)' = round1 || test '$(VARIANT)' = contrast
	$(MAKE) visible-bench PYTHON='$(PYTHON)' BENCH_ARGS='--models $(VARIANT) --seeds 0 --work experiments/visible/work/weights --output experiments/visible/work/weights/$(VARIANT).md'

weights:
	cd task1_watermark_localisation && { test -f "$(TRUSTMARK_STEM).pt" && test -f "$(TRUSTMARK_STEM).npz" || $(PYTHON) -m src.invisible_bench --work "$(INVISIBLE_WORK)" --initialize decoder; }
	cd task1_watermark_localisation && $(PYTHON) -m src.invisible_compare --work "$(INVISIBLE_WORK)" --maps "$(TRUSTMARK_STEM).npz"
	cd task1_watermark_localisation && $(PYTHON) -m src.invisible_product "$(TRUSTMARK_STEM).pt" "$(TRUSTMARK_STEM)-no-sam-stages.json" --resolution 128

task2:
	cd task2_model_equality && $(PYTHON) -m simlab.run $(SIM_ARGS)

task3:
	cd task3_unlearning && $(PYTHON) -m simlab.refit $(SIM_ARGS)

showcase:
	PYTHON='$(PYTHON)' bash showcase/make_showcase.sh
