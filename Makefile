SHELL := /bin/bash

PROJECT_ROOT := $(CURDIR)
BACKEND_DIR := $(PROJECT_ROOT)
VENV := $(PROJECT_ROOT)/.venv
PYTHON := $(VENV)/bin/python
PIP := $(VENV)/bin/pip
PYTEST := $(VENV)/bin/pytest

# Configuration
PORT := 8000
CONFIG_FILE := $(PROJECT_ROOT)/config/config.dev.yaml

.PHONY: help install dev test test-fast test-watch cov check format lint setup-hooks clean deb deb-clean

help:
	@echo "Machine Vision Backend - Development Commands"
	@echo "=============================================="
	@echo ""
	@echo "Setup:"
	@echo "  make install       Create venv and install all dependencies"
	@echo "  make setup-hooks   Install pre-commit hooks"
	@echo ""
	@echo "Development:"
	@echo "  make dev           Start with auto-reload (uvicorn --reload)"
	@echo ""
	@echo "Testing:"
	@echo "  make test          Run all tests"
	@echo "  make test-fast     Run fast tests only (skip slow)"
	@echo "  make test-watch    Watch mode for TDD (requires pytest-watch)"
	@echo "  make cov           Run tests with coverage report"
	@echo ""
	@echo "Code Quality:"
	@echo "  make format        Format code (black + isort)"
	@echo "  make lint          Lint code (flake8)"
	@echo "  make check         Run lint + tests (pre-push check)"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean         Remove caches and build artifacts"
	@echo "  make clean-all     Also remove venv"
	@echo ""
	@echo "Packaging:"
	@echo "  make deb           Build DEB package"
	@echo "  make deb-clean     Clean DEB build artifacts"

# ============================================
# Setup
# ============================================

install:
	@echo "Creating virtual environment..."
	@test -d $(VENV) || python3 -m venv $(VENV)
	@echo "Installing dependencies..."
	@$(PIP) install --upgrade pip
	@$(PIP) install -r $(PROJECT_ROOT)/requirements-dev.txt
	@echo ""
	@echo "✓ Installation complete!"
	@echo "  Activate venv: source $(VENV)/bin/activate"
	@echo "  Start dev server: make dev"

setup-hooks:
	@echo "Installing pre-commit hooks..."
	@test -d $(VENV) || (echo "Error: venv not found. Run 'make install' first." && exit 1)
	@cd $(PROJECT_ROOT) && $(PYTHON) -m pre_commit install
	@echo "✓ Pre-commit hooks installed!"

# ============================================
# Development
# ============================================

dev:
	@test -d $(VENV) || (echo "Error: venv not found. Run 'make install' first." && exit 1)
	@echo "Starting development server with auto-reload..."
	@echo "  Backend: http://localhost:$(PORT)"
	@echo "  API Docs: http://localhost:$(PORT)/docs"
	@echo ""
	@echo "Press Ctrl+C to stop"
	@echo ""
	@cd $(BACKEND_DIR) && \
		MV_CONFIG_FILE=$(CONFIG_FILE) \
		$(PYTHON) -m uvicorn main:app --reload --host=0.0.0.0 --port=$(PORT)

# ============================================
# Testing
# ============================================

test:
	@test -d $(VENV) || (echo "Error: venv not found. Run 'make install' first." && exit 1)
	@echo "Running all tests..."
	@cd $(PROJECT_ROOT) && PYTHONPATH=$(BACKEND_DIR) $(PYTEST) tests/ -v

test-fast:
	@test -d $(VENV) || (echo "Error: venv not found. Run 'make install' first." && exit 1)
	@echo "Running fast tests (skipping slow tests)..."
	@cd $(PROJECT_ROOT) && PYTHONPATH=$(BACKEND_DIR) $(PYTEST) tests/ -v -m "not slow"

test-watch:
	@test -d $(VENV) || (echo "Error: venv not found. Run 'make install' first." && exit 1)
	@echo "Starting test watch mode (TDD)..."
	@cd $(PROJECT_ROOT) && PYTHONPATH=$(BACKEND_DIR) $(PYTHON) -m pytest_watch tests/ -- -v

cov:
	@test -d $(VENV) || (echo "Error: venv not found. Run 'make install' first." && exit 1)
	@echo "Running tests with coverage..."
	@cd $(PROJECT_ROOT) && PYTHONPATH=$(BACKEND_DIR) $(PYTEST) tests/ --cov=src --cov-report=html --cov-report=term
	@echo ""
	@echo "✓ Coverage report generated: $(PROJECT_ROOT)/htmlcov/index.html"
	@command -v xdg-open >/dev/null 2>&1 && xdg-open $(PROJECT_ROOT)/htmlcov/index.html || true

# ============================================
# Code Quality
# ============================================

format:
	@test -d $(VENV) || (echo "Error: venv not found. Run 'make install' first." && exit 1)
	@echo "Formatting Python code..."
	@cd $(BACKEND_DIR) && $(PYTHON) -m black . --exclude .venv
	@cd $(BACKEND_DIR) && $(PYTHON) -m isort . --profile black --skip .venv
	@echo "✓ Code formatted!"

lint:
	@test -d $(VENV) || (echo "Error: venv not found. Run 'make install' first." && exit 1)
	@echo "Linting Python code..."
	@cd $(BACKEND_DIR) && $(PYTHON) -m flake8 . --exclude=.venv --max-line-length=100 --extend-ignore=E203,W503
	@echo "✓ Linting complete!"

check: lint test
	@echo ""
	@echo "✓ All checks passed!"

# ============================================
# Cleanup
# ============================================

clean:
	@echo "Cleaning build artifacts and caches..."
	@find $(BACKEND_DIR) -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find $(BACKEND_DIR) -type f -name "*.pyc" -delete 2>/dev/null || true
	@rm -rf $(PROJECT_ROOT)/.pytest_cache
	@rm -rf $(PROJECT_ROOT)/htmlcov
	@rm -rf $(PROJECT_ROOT)/.coverage
	@rm -rf $(PROJECT_ROOT)/var/log $(PROJECT_ROOT)/var/run
	@echo "✓ Cleanup complete!"

clean-all: clean
	@echo "Removing virtual environment..."
	@rm -rf $(VENV)
	@echo "✓ Deep cleanup complete!"

# ============================================
# Packaging
# ============================================

deb:
	@echo "Building DEB package..."
	@chmod +x $(PROJECT_ROOT)/packaging/build-deb.sh
	@$(PROJECT_ROOT)/packaging/build-deb.sh

deb-clean:
	@echo "Cleaning DEB build artifacts..."
	@rm -rf $(PROJECT_ROOT)/debian/.debhelper $(PROJECT_ROOT)/debian/machinevision
	@rm -f $(PROJECT_ROOT)/debian/files $(PROJECT_ROOT)/debian/*.substvars
	@rm -f $(PROJECT_ROOT)/debian/*.debhelper $(PROJECT_ROOT)/debian/*.log
	@rm -f $(PROJECT_ROOT)/debian/debhelper-build-stamp
	@rm -f $(PROJECT_ROOT)/../*.deb $(PROJECT_ROOT)/../*.changes $(PROJECT_ROOT)/../*.buildinfo
	@rm -rf $(PROJECT_ROOT)/build
	@echo "✓ DEB build artifacts cleaned!"
