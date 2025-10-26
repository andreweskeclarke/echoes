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
		echo "Creating conda environment from environment.yml..."; \
		conda env create -f environment.yml; \
	else \
		echo "Conda environment '$(CONDA_ENV)' already exists. Updating..."; \
		conda env update -f environment.yml --prune; \
	fi
	@echo "Setup complete! Run 'make check' to verify installation."

check:
	@$(CONDA_ACTIVATE) && ruff check --fix . && ruff format . && pytest tests/ && echo "All checks passed!"

test:
	@$(CONDA_ACTIVATE) && pytest tests/

deploy:
	@$(CONDA_ACTIVATE) && ./scripts/local_deploy.sh
