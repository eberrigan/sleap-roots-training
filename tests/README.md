# Tests for sleap-roots-training

This directory contains comprehensive tests for the sleap-roots-training package.

## Test Structure

- `test_config.py` - Tests for configuration management (`sleap_roots_training.config`)
- `test_train.py` - Tests for training orchestration (`sleap_roots_training.train`)
- `test_evaluate.py` - Tests for model evaluation (`sleap_roots_training.evaluate`)
- `test_models.py` - Tests for model artifact management (`sleap_roots_training.models`)
- `test_datasets.py` - Tests for dataset artifact management (`sleap_roots_training.datasets`)
- `test_imports.py` - Basic import verification tests
- `conftest.py` - Shared pytest fixtures and configuration

## Running Tests

### Quick Start

```bash
# Install with development dependencies
pip install -e .[dev]

# Run all tests with coverage
make test

# Run tests without coverage (faster)
make test-fast
```

### Detailed Commands

```bash
# Run all tests with coverage reports
pytest --cov=sleap_roots_training --cov-report=term-missing --cov-report=html

# Run specific test file
pytest tests/test_config.py -v

# Run tests with specific markers
pytest -m "unit" -v
pytest -m "integration" -v

# Run tests excluding slow ones
pytest -m "not slow" -v

# Run tests with specific keywords
pytest -k "test_config" -v
```

## Test Coverage

The test suite aims for high code coverage (80%+) to ensure code quality and reliability. Coverage reports are generated in multiple formats:

- **Terminal**: Shows coverage summary and missing lines
- **HTML**: Detailed interactive coverage report (`htmlcov/index.html`)
- **XML**: Machine-readable coverage data (`coverage.xml`)

## Test Configuration

Tests are configured using `pyproject.toml` with the following key settings:

- **Coverage threshold**: 80% minimum
- **Test discovery**: Automatic discovery of `test_*.py` files
- **Markers**: `unit`, `integration`, `slow` for test categorization
- **Warning filters**: Configured to handle expected warnings

## Fixtures

Common test fixtures are defined in `conftest.py`:

- `temp_dir` - Temporary directory for test files
- `mock_config` - Mock configuration dictionary
- `mock_wandb_run` - Mock W&B run object
- `mock_wandb_artifact` - Mock W&B artifact
- `sample_training_data` - Sample training DataFrame
- `sample_metrics` - Sample evaluation metrics
- `sample_sweep_config` - Sample sweep configuration

## CI/CD Integration

Tests are automatically run in GitHub Actions on:

- **Push/PR to main branch**
- **Multiple Python versions** (3.8, 3.9, 3.10, 3.11)  
- **Multiple operating systems** (Ubuntu, Windows, macOS)
- **Daily scheduled runs** for import testing

## Writing New Tests

When adding new functionality:

1. Create tests in the appropriate `test_*.py` file
2. Use descriptive test names starting with `test_`
3. Group related tests in test classes
4. Use fixtures from `conftest.py` where appropriate
5. Add appropriate markers (`@pytest.mark.unit`, etc.)
6. Mock external dependencies (W&B, SLEAP, file system)
7. Test both success and failure scenarios
8. Ensure tests are deterministic and don't depend on external state

## Common Issues

### Import Errors
- Ensure package is installed: `pip install -e .`
- Check that all dependencies are installed: `pip install -e .[dev]`

### Coverage Issues
- Use `# pragma: no cover` for lines that shouldn't be covered
- Check `[tool.coverage.report]` in `pyproject.toml` for exclusions

### Slow Tests
- Mark slow tests with `@pytest.mark.slow`
- Use `pytest -m "not slow"` to skip during development

### Mock Issues
- Check that mocks are properly configured in `conftest.py`
- Ensure mocks are reset between tests (handled automatically)