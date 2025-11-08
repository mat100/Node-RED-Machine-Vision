SHELL := /bin/bash

PROJECT_ROOT := $(CURDIR)
SCRIPTS_DIR := $(PROJECT_ROOT)/scripts
BACKEND_DIR := $(PROJECT_ROOT)/python-backend
BACKEND_VENV := $(BACKEND_DIR)/venv
BACKEND_PYTHON := $(BACKEND_VENV)/bin/python

# Runtime directories (fallback to project var/ for development)
LOG_DIR ?= $(PROJECT_ROOT)/var/log
RUN_DIR ?= $(PROJECT_ROOT)/var/run

BACKEND_PID_FILE := $(RUN_DIR)/backend.pid
BACKEND_LOG_FILE := $(LOG_DIR)/backend.log
PYTHON := python3
PORT_BACKEND := 8000

ARGS ?=

.PHONY: help install start stop status test clean reload logs format lint setup-hooks dev

help:
	@echo "Machine Vision Flow Backend - Core Commands"
	@echo "===================================="
	@echo ""
	@echo "  make install       Install Python dependencies"
	@echo "  make start         Start backend service"
	@echo "  make stop          Stop backend service"
	@echo "  make status        Show backend status"
	@echo "  make test          Run Python tests"
	@echo "  make clean         Clean runtime files (--all for everything)"
	@echo "  make reload        Reload backend service"
	@echo "  make logs          View backend logs"
	@echo ""
	@echo "Development (use VSCode for debugging - see ../.vscode/):"
	@echo "  make dev           Start with auto-reload (Python backend)"
	@echo "  make setup-hooks   Install pre-commit hooks"
	@echo "  make format        Format Python code (black, isort)"
	@echo "  make lint          Lint Python code (flake8)"

install:
	@echo "Installing Python backend dependencies..."
	@chmod +x $(SCRIPTS_DIR)/*.sh $(SCRIPTS_DIR)/lib/*.sh
	@$(SHELL) -lc "source \"$(SCRIPTS_DIR)/lib/services.sh\"; ensure_python_backend_env false"
	@echo "Installation complete!"

start:
	@$(SCRIPTS_DIR)/start.sh $(ARGS)

stop:
	@$(SCRIPTS_DIR)/stop.sh $(ARGS)

status:
	@$(SCRIPTS_DIR)/status.sh $(ARGS)

logs:
	@if [ -f $(BACKEND_LOG_FILE) ]; then \
		tail -f $(BACKEND_LOG_FILE) 2>/dev/null; \
	else \
		echo "No log file found. Start backend first with: make start"; \
	fi

clean:
	@echo "Cleaning runtime files..."
	@rm -rf $(LOG_DIR) $(RUN_DIR)
	@find $(BACKEND_DIR) -type d -name "__pycache__" -prune -exec rm -rf {} +
	@find $(PROJECT_ROOT) -name "*.pyc" -delete
ifeq ($(filter --all,$(MAKECMDGOALS)),--all)
	@echo "Cleaning test coverage, data, and dependencies..."
	@rm -rf $(BACKEND_DIR)/htmlcov $(BACKEND_DIR)/.coverage $(BACKEND_DIR)/.pytest_cache
	@find $(BACKEND_DIR)/data -mindepth 1 ! -name 'README.md' -delete 2>/dev/null || true
	@find $(BACKEND_DIR)/templates -mindepth 1 -delete 2>/dev/null || true
	@rm -rf $(BACKEND_DIR)/venv
	@echo "Complete cleanup done!"
else
	@echo "Runtime cleanup complete! (Use 'make clean --all' for full cleanup)"
endif

--all:
	@:

reload:
	@echo "Reloading services..."
	@$(MAKE) --no-print-directory stop
	@sleep 2
	@$(MAKE) --no-print-directory start

test:
	@echo "Running tests..."
	@$(SHELL) -lc "source \"$(SCRIPTS_DIR)/lib/services.sh\"; ensure_python_backend_env false"
	@if [ -d $(BACKEND_DIR)/tests ]; then \
		cd $(BACKEND_DIR) && $(BACKEND_PYTHON) -m pytest tests/ -v; \
	else \
		echo "No backend tests found in $(BACKEND_DIR)/tests"; \
	fi

setup-hooks:
	@echo "Installing pre-commit hooks..."
	@$(SHELL) -lc "source \"$(SCRIPTS_DIR)/lib/services.sh\"; ensure_python_backend_env false"
	@cd $(BACKEND_DIR) && $(BACKEND_PYTHON) -m pip install -q pre-commit black isort flake8
	@cd $(PROJECT_ROOT) && $(BACKEND_PYTHON) -m pre_commit install
	@echo "Pre-commit hooks installed! Run 'git commit' to trigger hooks."

format:
	@echo "Formatting Python code..."
	@cd $(BACKEND_DIR) && $(BACKEND_PYTHON) -m black . --exclude venv
	@cd $(BACKEND_DIR) && $(BACKEND_PYTHON) -m isort . --profile black --skip venv
	@echo "Code formatted!"

lint:
	@echo "Linting Python code..."
	@cd $(BACKEND_DIR) && $(BACKEND_PYTHON) -m flake8 . --exclude=venv --max-line-length=100 --extend-ignore=E203,W503
	@echo "Linting complete!"

dev:
	@echo "Starting Python backend development mode with auto-reload..."
	@echo ""
	@echo "Python backend: Auto-reloads on .py changes (uvicorn --reload)"
	@echo ""
	@echo "Press Ctrl+C to stop the service"
	@echo ""
	@cd $(BACKEND_DIR) && \
		export MV_CONFIG_FILE=$(BACKEND_DIR)/config.dev.yaml && \
		$(BACKEND_PYTHON) -m uvicorn main:app --reload --host=0.0.0.0 --port=$(PORT_BACKEND)
