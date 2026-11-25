.PHONY: install test lint experiments quick figures

install:
	pip install -e ".[dev]"

test:
	pytest -q

lint:
	ruff check minipath experiments tests

# Full runs, about 45 minutes on a 4-core machine. Run them one at a time:
# they are timing benchmarks and must not share the CPU.
experiments:
	python -m experiments.e1_dispatch
	python -m experiments.e2_gang
	python -m experiments.e3_pipeline
	python -m experiments.e4_fault_tolerance
	python -m experiments.e5_rl_overlap
	python -m experiments.plots

# Smoke-sized versions of everything, about 8 minutes.
quick:
	python -m experiments.e1_dispatch --quick
	python -m experiments.e2_gang --quick
	python -m experiments.e3_pipeline --quick
	python -m experiments.e4_fault_tolerance --quick
	python -m experiments.e5_rl_overlap --quick
	python -m experiments.plots

figures:
	python -m experiments.plots
