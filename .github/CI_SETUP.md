# CI/CD Setup Guide

This document explains our continuous integration setup, Python version strategy, and automated debugging system.

## Overview

Our CI/CD pipeline is designed to be **fast**, **reliable**, and **comprehensive** while handling the unique challenges of SLEAP integration.

## Python Version Strategy

We support **Python 3.7-3.9** for the package but use **Python 3.8** in CI for technical reasons:

| Installation Method | Python Version | Used Where |
|-------------------|---------------|------------|
| **Windows Conda SLEAP** | 3.7 | Local Development |
| **Linux/macOS Conda SLEAP** | 3.8 or 3.9 | Local Development |
| **PyPI SLEAP** (`sleap[pypi]==1.4.1`) | 3.8+ | CI/CD, Testing |

**Package Compatibility**: Python 3.7-3.9 to work with all SLEAP environments
**CI Choice**: Python 3.8 because:
- ✅ Works with PyPI SLEAP (required for CI)
- ✅ Avoids CUDA dependency conflicts in GitHub Actions
- ✅ Provides consistent testing environment
- ✅ Works across all CI platforms

### pyproject.toml Configuration

```toml
[project]
requires-python = ">=3.7,<3.10"
```

This ensures the package works with all SLEAP Python environments while maintaining compatibility.

## CI Workflows

### 1. Test Imports (`test-imports.yml`)

**Purpose**: Fast cross-platform import validation

```yaml
strategy:
  matrix:
    os: [ubuntu-latest, windows-latest, macos-latest]
    python-version: ["3.8"]
```

**What it does**:
- Installs SLEAP via PyPI with missing dependencies (`cattrs`, `attrs`)
- Tests basic imports across all platforms
- Validates package structure and metadata
- Runs exclusion tests to ensure proper packaging

**Why PyPI SLEAP**: Avoids CUDA dependency conflicts in GitHub Actions runners.

### 2. CI (`ci.yml`)

**Purpose**: Comprehensive integration testing with coverage

**Features**:
- **Testing**: Full pytest suite with 80% coverage threshold
- **Linting**: Code formatting checks with black
- **Building**: Package building and validation
- **Coverage**: Upload to Codecov with detailed reporting

**Dependencies**:
```bash
pip install sleap[pypi]==1.4.1
pip install -e .[dev]     # Our package with dev dependencies
```

## Installation Methods

### Local Development (Recommended)

**Follow official SLEAP installation instructions for your platform:**

**Windows/Linux:**
```bash
conda create -y -n sleap -c conda-forge -c nvidia -c sleap/label/dev -c sleap -c anaconda sleap=1.4.1
conda activate sleap
pip install -e .[dev]
```

**macOS:**
```bash
conda create -y -n sleap -c conda-forge -c anaconda -c sleap sleap=1.4.1
conda activate sleap
pip install -e .[dev]
```

> **Note**: The conda installation will provide the appropriate Python version (3.7-3.9) for your platform.

### CI/Testing Environment

```bash
# Install PyPI SLEAP
pip install sleap[pypi]==1.4.1

# Install our package
pip install -e .[dev]
```

## Testing Strategy

### Test Structure

```
tests/
├── test_config.py      # Configuration management tests
├── test_train.py       # Training orchestration tests  
├── test_evaluate.py    # Model evaluation tests
├── test_models.py      # Model artifact tests
├── test_datasets.py    # Dataset artifact tests
├── test_imports.py     # Import validation tests
└── conftest.py         # Shared test fixtures
```

### Coverage Requirements

- **Target**: 80% code coverage
- **Enforcement**: CI fails if coverage drops below threshold
- **Reporting**: Codecov integration with PR comments

### Mocking Strategy

Our tests use comprehensive mocking for:
- **SLEAP operations**: Model loading, training, evaluation
- **W&B operations**: Run initialization, artifact logging
- **File system**: Config files, model directories
- **External APIs**: All network calls

## Troubleshooting

### Common Issues

1. **Python Version Conflicts**
   - **Cause**: Using incompatible Python version (not 3.7-3.9)
   - **Fix**: Ensure your environment uses Python 3.7-3.9; CI uses 3.8

2. **CUDA Dependency Errors**
   - **Cause**: Conda SLEAP trying to install CUDA in CI
   - **Fix**: Use PyPI SLEAP in CI, conda locally

3. **Import Errors**
   - **Cause**: Missing dependencies or incorrect installation order
   - **Fix**: Follow the exact installation sequence in CI workflows

## Best Practices

### For Developers

1. **Follow SLEAP installation instructions** for your platform (installs appropriate Python version)
2. **Use conda SLEAP locally** for full functionality and GPU support
3. **Don't worry about Python version** - the package supports 3.7-3.9
4. **Run formatting checks** before pushing
5. **Monitor coverage** and add tests for new features

### For CI Maintenance

1. **Keep dependencies minimal** in CI workflows
2. **Use caching** for pip packages where possible
3. **Update action versions** regularly
4. **Monitor auto-debug effectiveness** and improve prompts

## Badges

Current CI status badges:

```markdown
[![Test Imports](https://github.com/eberrigan/sleap-roots-training/workflows/Test%20Imports/badge.svg)](https://github.com/eberrigan/sleap-roots-training/actions/workflows/test-imports.yml)
[![CI](https://github.com/eberrigan/sleap-roots-training/workflows/CI/badge.svg)](https://github.com/eberrigan/sleap-roots-training/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/eberrigan/sleap-roots-training/branch/main/graph/badge.svg)](https://codecov.io/gh/eberrigan/sleap-roots-training)
```

## Future Improvements

1. **Multi-Python Testing**: Test across Python 3.7-3.9 in CI (currently standardized on 3.8)
2. **Performance Benchmarks**: Add performance regression testing
3. **Integration Tests**: Add end-to-end training pipeline tests
4. **Documentation**: Auto-generate API documentation
5. **Security**: Add dependency security scanning

This setup ensures reliable, fast CI/CD while handling the complexities of SLEAP integration and supporting all SLEAP Python environments.