.PHONY: install switch test lint clean format update

# Set the default model for installation
MODEL ?= 0.5b

install:
	@echo "Installing Jev MCP Environment..."
	./jev_mac_manager.sh install $(MODEL)

switch:
	@echo "Switching Model..."
	./jev_mac_manager.sh switch $(MODEL)

update:
	@echo "Updating Jev MCP Environment..."
	./jev_mac_manager.sh update

format:
	@echo "Formatting code with Ruff..."
	bash -c "source .venv/bin/activate && ruff format ."

lint:
	@echo "Running MyPy Static Type Checking..."
	bash -c "source .venv/bin/activate && mypy src/jev_mcp/"

test:
	@echo "Running Pytest Mathematical Verification..."
	bash -c "source .venv/bin/activate && pytest tests/"

clean:
	@echo "Completely uninstalling Jev MCP artifacts..."
	./jev_mac_manager.sh uninstall --headless
