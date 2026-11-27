PYTHON ?= python
BENCH_ARGS ?=
VARIANT ?= round1
SIM_ARGS ?=

.DEFAULT_GOAL := help
.PHONY: help install test lint bench bench-audit shipped weights task2 task3 showcase

help:
	@printf '%s\n' 'install test lint bench bench-audit shipped weights task2 task3 showcase'

install:
	$(PYTHON) -m pip install -r benchmarks/requirements.txt -e '.[app,onnx,dev,showcase]'
	$(PYTHON) -m playwright install chromium

test:
	cd task1_watermark_localisation && $(PYTHON) -m pytest -q tests
	cd task2_model_equality && $(PYTHON) -m pytest -q tests
	cd task3_unlearning && $(PYTHON) -m pytest -q tests

lint:
	$(PYTHON) -m ruff check .
	$(PYTHON) -m mypy

bench:
	cd task1_watermark_localisation && $(PYTHON) -m src.bench $(BENCH_ARGS)

bench-audit:
	cd task1_watermark_localisation && $(PYTHON) -m src.audit_watermarks

shipped:
	cd task1_watermark_localisation && $(PYTHON) -m src.shipped_report

weights:
	@test '$(VARIANT)' = round1 || test '$(VARIANT)' = contrast
	$(MAKE) bench PYTHON='$(PYTHON)' BENCH_ARGS='--models $(VARIANT) --seeds 0 --work work/weights --output work/weights/$(VARIANT).md'

task2:
	cd task2_model_equality && $(PYTHON) -m simlab.run $(SIM_ARGS)

task3:
	cd task3_unlearning && $(PYTHON) -m simlab.refit $(SIM_ARGS)

showcase:
	PYTHON='$(PYTHON)' bash showcase/make_showcase.sh
