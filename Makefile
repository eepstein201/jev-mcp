.PHONY: install test lint clean format update start stop hybrid

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
	bash -c "source .venv/bin/activate && ruff format ."

lint:
	@echo "Running MyPy Static Type Checking..."
	bash -c "source .venv/bin/activate && mypy src/jev_mcp/"

test:
	@echo "Running Pytest with Strict 85% Code Coverage Requirement..."
	bash -c "source .venv/bin/activate && pytest --cov=src --cov-report=term-missing --cov-fail-under=85 tests/"

clean:
	@echo "Completely uninstalling Jev MCP artifacts..."
	./jev_mac_manager.sh uninstall --headless

hybrid:
	@echo "Starting Dual-Engine Hybrid Mode (Kev 0.8B + Qwen 7B)..."
	./jev_mac_manager.sh hybrid
