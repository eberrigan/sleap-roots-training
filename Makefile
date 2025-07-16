# Makefile for sleap-roots-training

.PHONY: help install install-dev test test-cov test-fast lint format clean build docs

help:  ## Show this help message
	@echo "Available targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-20s %s\n", $$1, $$2}'

install:  ## Install package in development mode
	pip install -e .

install-dev:  ## Install package with development dependencies
	pip install -e .[dev]

test:  ## Run all tests with coverage
	pytest --cov=sleap_roots_training --cov-report=term-missing --cov-report=html --cov-report=xml

test-cov:  ## Run tests with coverage and open HTML report
	pytest --cov=sleap_roots_training --cov-report=html --cov-report=term-missing
	@echo "Opening coverage report..."
	@python -c "import webbrowser; webbrowser.open('htmlcov/index.html')"

test-fast:  ## Run tests without coverage (faster)
	pytest -v

test-unit:  ## Run only unit tests
	pytest -v -m "unit"

test-integration:  ## Run only integration tests
	pytest -v -m "integration"

test-imports:  ## Test that all modules can be imported
	pytest tests/test_imports.py -v

lint:  ## Run linting checks
	black --check sleap_roots_training/ tests/

format:  ## Format code with black
	black sleap_roots_training/ tests/

clean:  ## Clean up build artifacts and cache
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf htmlcov/
	rm -f coverage.xml
	rm -f .coverage
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete

build:  ## Build package
	python -m build

docs:  ## Generate documentation (placeholder)
	@echo "Documentation generation not yet implemented"

check:  ## Run all checks (lint, test, build)
	@echo "Running linting..."
	$(MAKE) lint
	@echo "Running tests..."
	$(MAKE) test
	@echo "Building package..."
	$(MAKE) build
	@echo "All checks passed!"

ci:  ## Run CI pipeline locally
	@echo "Running CI pipeline..."
	$(MAKE) format
	$(MAKE) test
	$(MAKE) build
	@echo "CI pipeline completed!"

# Coverage targets
coverage-html:  ## Generate HTML coverage report
	pytest --cov=sleap_roots_training --cov-report=html
	@echo "Coverage report generated in htmlcov/"

coverage-xml:  ## Generate XML coverage report
	pytest --cov=sleap_roots_training --cov-report=xml
	@echo "Coverage report generated in coverage.xml"

coverage-term:  ## Show coverage in terminal
	pytest --cov=sleap_roots_training --cov-report=term-missing

# Test specific modules
test-config:  ## Test config module
	pytest tests/test_config.py -v

test-train:  ## Test train module
	pytest tests/test_train.py -v

test-evaluate:  ## Test evaluate module  
	pytest tests/test_evaluate.py -v

test-models:  ## Test models module
	pytest tests/test_models.py -v

test-datasets:  ## Test datasets module
	pytest tests/test_datasets.py -v