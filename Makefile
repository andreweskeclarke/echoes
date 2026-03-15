SHELL := /bin/bash
.PHONY: check test deploy help setup services services-stop services-status

CONDA_ENV := echoes
CONDA_ACTIVATE := source $$(conda info --base)/etc/profile.d/conda.sh && conda activate $(CONDA_ENV)

DATA_DIR := /mnt/echoes_data
CONDA_BIN := $$(conda info --base)/envs/$(CONDA_ENV)/bin

help:
	@echo "Available commands:"
	@echo "  make setup          - Set up conda environment and install dependencies (run once)"
	@echo "  make check          - Run all code quality checks (lint, format, test)"
	@echo "  make test           - Run pytest tests"
	@echo "  make services       - Start MLflow, TensorBoard, and dashboard in the background"
	@echo "  make services-stop  - Stop background services"
	@echo "  make services-status - Show running service PIDs"
	@echo "  make deploy         - Full production deploy via systemd/nginx (requires sudo)"

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

services:
	@echo "Starting MLflow, TensorBoard, and dashboard..."
	@mkdir -p $(DATA_DIR)/mlruns $(DATA_DIR)/tfruns
	@$(CONDA_BIN)/mlflow ui --host 0.0.0.0 --port 5000 \
		--backend-store-uri $(DATA_DIR)/mlruns \
		>/tmp/mlflow.log 2>&1 & echo $$! > /tmp/mlflow.pid
	@$(CONDA_BIN)/tensorboard --logdir=$(DATA_DIR)/tfruns \
		--host=0.0.0.0 --port=6006 \
		>/tmp/tensorboard.log 2>&1 & echo $$! > /tmp/tensorboard.pid
	@echo "Services started:"
	@echo "  MLflow:      http://localhost:5000  (log: /tmp/mlflow.log)"
	@echo "  TensorBoard: http://localhost:6006  (log: /tmp/tensorboard.log)"

services-stop:
	@for svc in mlflow tensorboard; do \
		if [ -f /tmp/$$svc.pid ]; then \
			kill $$(cat /tmp/$$svc.pid) 2>/dev/null && echo "Stopped $$svc" || echo "$$svc not running"; \
			rm -f /tmp/$$svc.pid; \
		fi; \
	done

services-status:
	@for svc in mlflow tensorboard; do \
		if [ -f /tmp/$$svc.pid ] && kill -0 $$(cat /tmp/$$svc.pid) 2>/dev/null; then \
			echo "$$svc: running (PID $$(cat /tmp/$$svc.pid))"; \
		else \
			echo "$$svc: stopped"; \
		fi; \
	done

deploy:
	@echo "Deploying services (requires sudo)..."
	@sudo bash -c "$(CONDA_ACTIVATE) && ./scripts/local_deploy.sh"
