.PHONY: install test lint clean format format-check audit check update start stop hybrid train eval eval-browser eval-triage eval-usage

install:
	@echo "Installing Jev MCP Environment..."
	./jev_mac_manager.sh install
	./jev_mac_manager.sh hybrid

update:
	@echo "Updating Jev MCP Environment..."
	./jev_mac_manager.sh update

start:
	@echo "Starting Jev MCP Daemon..."
	./jev_mac_manager.sh hybrid

stop:
	@echo "Stopping Jev MCP Daemon..."
	./jev_mac_manager.sh stop

format:
	@echo "Formatting code with Ruff..."
	bash -c "source .venv/bin/activate && ruff format src tests"

format-check:
	@echo "Checking formatting with Ruff..."
	bash -c "source .venv/bin/activate && ruff format --check src tests"

audit:
	@echo "Auditing installed packages for known advisories (accepted ones: .pip-audit-ignore)..."
	bash -c "source .venv/bin/activate && ./scripts/pip_audit.sh"

check: format-check lint test audit

lint:
	@echo "Running MyPy Static Type Checking..."
	bash -c "source .venv/bin/activate && mypy src/jev_mcp/"

test:
	@echo "Running Pytest with Strict 85% Code Coverage Requirement..."
	bash -c "source .venv/bin/activate && pytest --cov=src --cov-report=term-missing --cov-fail-under=85 tests/"

eval:
	@echo "Benchmarking Kev 0.8B and Qwen 7B on tests/golden_dataset.json (daemons must be running)..."
	bash -c "source .venv/bin/activate && python tests/run_evals.py"

eval-browser:
	@echo "Running each browser-agent case several times and reporting pass rates (daemons must be running; drives a real browser)..."
	bash -c "source .venv/bin/activate && python -m tests.evals.browser_eval"

DATASET ?= tests/evals/triage_synthetic.jsonl

eval-triage:
	@echo "Triaging every email in $(DATASET) and reporting agreement with its expected routing (daemons must be running)..."
	bash -c "source .venv/bin/activate && python -m tests.evals.triage_eval --dataset $(DATASET)"

# The runner is the mcp-builder skill's scripts/evaluation.py. It needs `anthropic` and an `mcp` 1.x client,
# so it runs from its own venv (see tests/evals/usage/README.md), not the project's.
USAGE_EVAL_RUNNER ?=
USAGE_EVAL_PYTHON ?= $(HOME)/.jev/usage-eval-venv/bin/python
USAGE_EVAL_MODEL ?= claude-sonnet-5-5
USAGE_EVAL_HOME = /tmp/jev-usage-eval-home

eval-usage:
	@test -n "$(USAGE_EVAL_RUNNER)" || { echo "Set USAGE_EVAL_RUNNER to the mcp-builder skill's scripts/evaluation.py"; exit 1; }
	@test -n "$$ANTHROPIC_API_KEY" || { echo "Set ANTHROPIC_API_KEY: the runner calls the Anthropic API"; exit 1; }
	@echo "Asking $(USAGE_EVAL_MODEL) the 10 usage questions against a seeded home (daemons must be running; calls the Anthropic API)..."
	.venv/bin/python tests/evals/usage/seed_home.py
	$(USAGE_EVAL_PYTHON) tests/evals/usage/run_eval.py $(USAGE_EVAL_RUNNER) -t stdio -m $(USAGE_EVAL_MODEL) \
		-c $(CURDIR)/.venv/bin/jev-mcp \
		-e HOME=$(USAGE_EVAL_HOME) JEV_ALLOWED_ROOTS=$(USAGE_EVAL_HOME)/fixture PYTHONPATH=$(CURDIR)/src \
		tests/evals/usage/evaluation.xml

clean:
	@echo "Completely uninstalling Jev MCP artifacts..."
	./jev_mac_manager.sh uninstall --headless

hybrid:
	@echo "Starting Dual-Engine Hybrid Mode (Kev 0.8B + Qwen 7B)..."
	./jev_mac_manager.sh hybrid

train:
	@echo "Local LoRA Training is managed via the MCP tool interface."
	@echo "To trigger training, use the '/jev-mcp:train' command in your IDE."
	@echo "By default, this will use MLX for training and llama.cpp for auto-fusion."
