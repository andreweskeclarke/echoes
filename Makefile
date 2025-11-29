SHELL := /bin/bash
.PHONY: check test deploy help setup

CONDA_ENV := echoes
CONDA_ACTIVATE := source $$(conda info --base)/etc/profile.d/conda.sh && conda activate $(CONDA_ENV)

help:
	@echo "Available commands:"
	@echo "  make setup       - Set up conda environment and install dependencies (run once)"
	@echo "  make check       - Run all code quality checks (lint, format, test)"
	@echo "  make test        - Run pytest tests"
	@echo "  make deploy      - Deploy services with local_deploy.sh"

setup:
	@echo "Setting up Echoes development environment..."
	@if ! conda env list | grep -q "$(CONDA_ENV)"; then \
		echo "Creating conda environment..."; \
		conda create -n $(CONDA_ENV) python=3.11 -y; \
	fi
	@echo "Installing dependencies with pip..."
	@$(CONDA_ACTIVATE) && pip install -r requirements.txt
	@echo "Setup complete! Run 'make check' to verify installation."

check:
	@echo "Running code quality checks..."
	@echo ""
	@echo "Step 1: Running ruff linter (with auto-fix)..."
	@$(CONDA_ACTIVATE) && ruff check --fix .
	@echo ""
	@echo "Step 2: Formatting code with ruff..."
	@$(CONDA_ACTIVATE) && ruff format .
	@echo ""
	@echo "Step 3: Running pytest..."
	@$(CONDA_ACTIVATE) && (pytest tests/ || [ $$? -eq 5 ])
	@echo ""
	@echo "✓ All checks passed!"

test:
	@$(CONDA_ACTIVATE) && pytest tests/

deploy:
	@echo "Deploying services (requires sudo)..."
	@sudo bash -c "$(CONDA_ACTIVATE) && ./scripts/local_deploy.sh"
