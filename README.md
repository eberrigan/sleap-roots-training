# sleap-roots-training

[![Branch Checks](https://github.com/eberrigan/sleap-roots-training/workflows/Branch%20Checks/badge.svg)](https://github.com/eberrigan/sleap-roots-training/actions/workflows/branch-checks.yml)
[![Test Imports](https://github.com/eberrigan/sleap-roots-training/workflows/Test%20Imports/badge.svg)](https://github.com/eberrigan/sleap-roots-training/actions/workflows/test-imports.yml)
[![CI with Conda](https://github.com/eberrigan/sleap-roots-training/workflows/CI%20with%20Conda/badge.svg)](https://github.com/eberrigan/sleap-roots-training/actions/workflows/ci-conda.yml)
[![CI](https://github.com/eberrigan/sleap-roots-training/workflows/CI/badge.svg)](https://github.com/eberrigan/sleap-roots-training/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/eberrigan/sleap-roots-training/branch/main/graph/badge.svg)](https://codecov.io/gh/eberrigan/sleap-roots-training)

A Python package for training and evaluating SLEAP models for root tracking, with integrated Weights & Biases logging and experiment management.

## Features

- **SLEAP Integration**: Seamless integration with SLEAP pose estimation framework
- **W&B Logging**: Comprehensive experiment tracking with Weights & Biases
- **Model Management**: Automated model artifact creation and registry management
- **Evaluation Tools**: Built-in model evaluation and visualization capabilities
- **Sweep Support**: Parameter sweep capabilities for hyperparameter optimization
- **Jupyter Notebooks**: Interactive notebooks for training and analysis

## Installation

### Prerequisites

1. **Python 3.7-3.9**: SLEAP requires Python 3.7, 3.8, or 3.9 (not 3.10+)
2. **SLEAP Environment**: Install SLEAP following the [official guide](https://sleap.ai/installation.html)
3. **Weights & Biases**: Sign up for a [W&B account](https://wandb.ai)

### Quick Setup

#### Step 1: Install SLEAP

**Windows/Linux:**
```bash
conda create -y -n sleap -c conda-forge -c nvidia -c sleap/label/dev -c sleap -c anaconda sleap=1.4.1
conda activate sleap
```

**macOS:**
```bash
conda create -y -n sleap -c conda-forge -c anaconda -c sleap sleap=1.4.1
conda activate sleap
```

**PyPI (alternative):**
```bash
pip install sleap[pypi]==1.4.1
```

#### Step 2: Install this package

```bash
# Clone and install this package
git clone https://github.com/eberrigan/sleap-roots-training.git
cd sleap-roots-training
pip install -e .[dev]

# Login to W&B
wandb login

# Verify installation
python verify_installation.py
```

## Usage

### Configuration

The package uses a YAML configuration file for W&B settings:

```python
from sleap_roots_training.config import load_config, update_config

# Load current configuration
config = load_config()

# Update configuration
update_config(
    experiment_name="my_experiment",
    registry="my_registry"
)
```

### Training

```python
from sleap_roots_training.train import main

# Run training with configuration
main(
    csv_path="path/to/train_test_splits.csv",
    tags=["experiment", "tag"],
    model_tags=["model", "tag"],
    use_sweep=False,
    link_to_registry=True
)
```

### Evaluation

```python
from sleap_roots_training.evaluate import evaluate_model

# Evaluate a model
labels_pr, metrics, metrics_summary = evaluate_model(
    model_artifact_name="my_model_v001",
    test_artifact_name="my_test_data",
    output_dir="evaluation_results"
)
```

## Development

### Testing

```bash
# Run all tests with coverage
make test

# Run tests without coverage (faster)
make test-fast

# Run specific test modules
pytest tests/test_config.py -v

# Run tests with specific markers
pytest -m "unit" -v
```

### Code Quality

```bash
# Format code
make format

# Check formatting
make lint

# Run full CI pipeline locally
make ci
```

### Project Structure

```
sleap-roots-training/
├── sleap_roots_training/          # Main package
│   ├── config.py                  # Configuration management
│   ├── train.py                   # Training orchestration
│   ├── evaluate.py                # Model evaluation
│   ├── models.py                  # Model artifact management
│   └── datasets.py                # Dataset artifact management
├── tests/                         # Test suite
├── helper_notebooks/              # Reusable notebook templates
├── *.ipynb                        # Experiment notebooks
└── pyproject.toml                 # Package configuration
```

## Notebooks

The repository includes Jupyter notebooks for interactive experimentation:

- **Helper Notebooks**: Reusable templates in `helper_notebooks/`
- **Experiment Notebooks**: Date-stamped experiments (e.g., `20250711_experiment_name.ipynb`)

### Notebook Best Practices

1. **Run from repository root** for proper imports
2. **Save copies** of helper notebooks with experiment-specific names
3. **Use separate branches** for different experiments
4. **Follow naming convention**: `YYYYMMDD_experiment_description.ipynb`

## CI/CD

The project uses multiple GitHub Actions workflows for comprehensive testing:

- **Branch Checks**: Fast validation (formatting, imports, basic tests)
- **Test Imports**: Cross-platform compatibility testing
- **CI with Conda**: SLEAP integration testing with conda
- **CI**: Full integration testing with comprehensive coverage

All workflows run on every push and pull request to ensure code quality.

## Documentation

- **[CLAUDE.md](CLAUDE.md)**: Comprehensive development guide
- **[tests/README.md](tests/README.md)**: Testing documentation
- **[STATUS_BADGES.md](STATUS_BADGES.md)**: CI workflow guide
- **[Notion Guide](https://www.notion.so/SLEAP-Roots-Model-Training-Guide-21b4a67a766780ec91c0f92ae459ef2c?source=copy_link)**: Complete training guide

## Contributing

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/my-feature`
3. **Make changes** and add tests
4. **Ensure tests pass**: `make test`
5. **Check formatting**: `make lint`
6. **Push branch** and create a pull request
7. **Ensure all CI checks pass**

## License

This project is licensed under the terms specified in the [LICENSE](LICENSE) file.

## Support

For questions and support:
- **Issues**: [GitHub Issues](https://github.com/eberrigan/sleap-roots-training/issues)
- **Documentation**: See [CLAUDE.md](CLAUDE.md) for detailed development guide
- **SLEAP Support**: [SLEAP Documentation](https://sleap.ai)