.PHONY: help install test lint format clean run-pipeline
CODE_DIRS = src/ tests/

help: ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

setup-hooks: ## Install pre-commit hooks
	uv run pre-commit install
	@echo "Pre-commit hooks installed!

install: ## Install dependencies
	uv sync --group dev
	uv run pre-commit install

pre-commit: ## Run pre-commit on all files
	uv run pre-commit run --all-files

test: ## Run tests
	uv run pytest tests/ -v

test-cov: ## Run tests with coverage
	uv run pytest tests/ -v --cov=src --cov-report=term-missing --cov-report=html

format: ## Format code using Ruff formatter
	@echo "Formatting code..."
	uv run ruff format $(CODE_DIRS)

format-check: ## Check if code is formatted correctly (for CI)
	@echo "Checking code formatting..."
	uv run ruff format --check $(CODE_DIRS)

lint: ## Check for linting issues using Ruff (no fixes)
	@echo "Linting code..."
	uv run ruff check $(CODE_DIRS)

lint-fix: ## Attempt to automatically fix linting issues using Ruff
	@echo "Attempting to fix linting issues..."
	uv run ruff check --fix $(CODE_DIRS)

autofix: lint-fix format ## Automatically fix linting issues and then format code
	@echo "Code auto-fixing and formatting complete."

type-check: ## Run mypy type checking separately
	uv run mypy src/ --config-file=pyproject.toml

check-all: lint format-check type-check test pre-commit ## Run all checks (lint + test)
	@echo "All checks complete."

clean: ## Clean up generated files
	rm -rf .pytest_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	rm -rf .mypy_cache/
	rm -rf __pycache__/
	find . -type d -name "__pycache__" -delete
	find . -type f -name "*.pyc" -delete


clean-data: ## Clean all files in data/ directory (except .gitignore)
	@echo "Cleaning data directory..."
	find data/ -type f ! -name ".gitignore" -delete
	@echo "Data directory cleaned (kept .gitignore)"

clean-models: ## Clean all files in models/ directory (except .gitignore)
	@echo "Cleaning models directory..."
	find models/ -type f ! -name ".gitignore" -delete
	@echo "Models directory cleaned (kept .gitignore)"

clean-all: clean clean-data clean-models ## Clean everything (dev files, data, and models)
	@echo "Complete cleanup finished!"

run-pipeline: ## Run the complete ML pipeline
	python src/load_data.py
	python src/split_dataset.py --test_size 0.2
	python src/train.py

run-evaluation:
	python src/evaluate.py
