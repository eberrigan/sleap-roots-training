# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Python package for training and evaluating SLEAP (Social LEAP Estimates Animal Poses) models for root tracking, with integrated Weights & Biases (W&B) logging. The codebase provides a wrapper around SLEAP and W&B for model training, evaluation, and experiment management.

## Key Dependencies

- `sleap` - Core pose estimation library
- `wandb` - Experiment tracking and model management
- `jupyterlab` - For interactive notebooks
- `matplotlib`, `seaborn` - Visualization
- `pandas`, `numpy` - Data manipulation

## Development Setup

### 1. Install SLEAP

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

### 2. Setup Development Environment

```bash
# Install development dependencies
pip install -e .[dev]

# Login to W&B
wandb login
```

### 3. Environment Activation (Windows)

For running tests and development commands on Windows, use:
```bash
cd "C:\repos\sleap-roots-training" && source /c/Users/Elizabeth/miniforge3/etc/profile.d/conda.sh && conda activate sleap_v1.4.1
```

### 3. Development Notes

- Work from repository root so `sleap_roots_training` imports work correctly
- Use separate branches for different experiments
- Follow the testing guidelines in this document

## Common Commands

### Installation
```bash
pip install -e .[dev]  # Install in development mode
```

### Testing
```bash
make test              # Run all tests with coverage
make test-fast         # Run tests without coverage (faster)
make test-unit         # Run only unit tests
make test-imports      # Test imports only
pytest tests/test_config.py -v  # Test specific module
```

### Development Tools
```bash
make format           # Format code with black
make lint            # Check code formatting
make clean           # Clean build artifacts
make build           # Build package
make ci              # Run full CI pipeline locally

# Manual formatting (when make is not available)
python -m black <file_paths>  # Format specific files
python -m black tests/        # Format all test files
```

## Architecture

### Core Modules

- **`sleap_roots_training/config.py`**: Configuration management with YAML file support. Handles W&B project settings, experiment names, and registry configuration.

- **`sleap_roots_training/train.py`**: Main training orchestration. Contains the primary `main()` function that processes training runs, handles W&B logging, and manages model artifacts. Supports both single training runs and parameter sweeps.

- **`sleap_roots_training/models.py`**: Model artifact management. Functions for fetching, linking, and promoting models in W&B registries.

- **`sleap_roots_training/evaluate.py`**: Model evaluation and visualization. Contains functions for generating predictions, creating visualizations, and evaluating model performance against test datasets.

- **`sleap_roots_training/datasets.py`**: Dataset artifact creation and management for W&B.

### Configuration System

The configuration is managed through `config.yaml` in the main module directory. Key configuration parameters:
- `project_name`: W&B project name
- `entity_name`: W&B entity/organization  
- `experiment_name`: Current experiment identifier
- `registry`: W&B model registry name
- `collection_name`: Registry collection name

Configuration can be updated programmatically using functions in `config.py`.

### Training Workflow

1. **Data Preparation**: Train/test splits are managed via CSV files containing paths to configuration files
2. **Configuration**: Each training version has an `initial_config_modified_v00{version}.json` file
3. **Training Execution**: Uses `sleap-train` command with configuration files
4. **Artifact Logging**: Models are logged to W&B with evaluation metrics and visualizations
5. **Registry Management**: Models can be automatically linked to W&B model registries

### Notebook Integration

The repository contains numerous Jupyter notebooks following naming patterns:
- `YYYYMMDD_experiment_description.ipynb` - Main experiment notebooks
- `helper_notebooks/` - Reusable notebook templates

Always save copies of helper notebooks with experiment-specific names and work on separate branches.

## Key Functions

### Training (`train.py`)
- `main()`: Main entry point for training runs
- `run_single_training()`: Execute single training run
- `run_sweep_training()`: Execute W&B parameter sweeps
- `log_model_artifact_with_evals()`: Log trained models with evaluations

### Evaluation (`evaluate.py`)
- `evaluate_model()`: Evaluate model against test dataset
- `fetch_sweep_metrics()`: Retrieve metrics from W&B sweeps
- `predictions_viz()`: Generate prediction visualizations
- `fetch_metrics_from_sweep_pattern()`: **[NEW]** Find and fetch metrics from sweeps by name pattern
- `group_sweep_runs_retroactively()`: **[NEW]** Retroactively group sweep runs for organization

### Configuration (`config.py`)
- `load_config()`: Load configuration from YAML
- `update_config()`: Update specific configuration values
- `CONFIG`: Global configuration dictionary

## Testing

Comprehensive test suite with high code coverage (target: 80%+) using pytest:

### Running Tests

```bash
# Run all tests with coverage
pytest --cov=sleap_roots_training --cov-report=term-missing --cov-report=html

# Run tests without coverage (faster)
pytest -v

# Run specific test file
pytest tests/test_config.py -v

# Run tests with specific markers
pytest -m "unit" -v
pytest -m "integration" -v

# Using Makefile shortcuts
make test          # Run all tests with coverage
make test-fast     # Run tests without coverage
make test-unit     # Run only unit tests
make test-imports  # Test imports only
```

### Test Structure

**Test Organization Guidelines:**
- **One-to-one mapping**: For every module `sleap_roots_training/<module>.py`, there is a corresponding test file `tests/test_<module>.py`
- **Centralized fixtures**: All fixtures are defined in `tests/fixtures.py` and imported by test modules
- **Real test data**: Test data is stored in `tests/data/` directory with actual SLEAP experiment files

**Test Files:**
- `tests/test_config.py` - Configuration management tests
- `tests/test_train.py` - Training workflow tests (unit tests with mocking)
- `tests/test_evaluate.py` - Evaluation and metrics tests  
- `tests/test_models.py` - Model artifact management tests
- `tests/test_datasets.py` - Dataset artifact tests
- `tests/test_sweep_integration.py` - Sweep integration tests with real data
- `tests/test_imports.py` - Basic import verification
- `tests/conftest.py` - Shared fixtures and test configuration
- `tests/fixtures.py` - Reusable test fixtures for real data
- `tests/data/` - Real test data including SLEAP experiment files

### Test Fixtures

**Reusable fixtures are defined in `tests/fixtures.py` for use across all test modules:**

- `sweep_experiment_data` - Real SLEAP experiment data with CSV, config, and SLEAP files
- `temp_experiment_dir` - Temporary copy of experiment data for safe testing
- `realistic_sweep_config` - Full W&B sweep configuration with multiple parameters
- `small_sweep_config` - Minimal sweep configuration for faster testing
- `mock_models_dir` - Mock directory structure for testing model discovery
- `environment_config` - Test environment configuration values

**Usage in tests:**
```python
# Import fixtures at top of test file
from tests.fixtures import sweep_experiment_data, temp_experiment_dir

# Use fixtures in test functions
def test_my_function(sweep_experiment_data, temp_experiment_dir):
    # Access real SLEAP data
    config = sweep_experiment_data["config"]
    df = sweep_experiment_data["df"]
    
    # Use temporary directory for safe testing
    temp_config = temp_experiment_dir["config"]
    temp_csv = temp_experiment_dir["csv_path"]
```

**Cross-platform compatibility:**
- All fixtures handle Windows/Linux/macOS path differences
- Use forward slashes in paths to avoid Windows backslash issues
- Temporary directories are automatically cleaned up after tests

### Test Coverage

- Code coverage is measured and reported for all modules
- Minimum coverage threshold: 80%
- Coverage reports generated in HTML format (`htmlcov/`)
- XML coverage reports for CI integration (`coverage.xml`)

### Test Development Workflow

When developing or modifying tests, follow this workflow:

1. **Activate environment**: Use the correct conda environment activation
   ```bash
   cd "C:\repos\sleap-roots-training" && source /c/Users/Elizabeth/miniforge3/etc/profile.d/conda.sh && conda activate sleap_v1.4.1
   ```

2. **Run tests**: Execute tests to check current status
   ```bash
   python -m pytest --cov=sleap_roots_training --cov-report=term-missing tests/test_<module>.py
   ```

3. **Format code**: Always format test files before committing
   ```bash
   python -m black tests/test_<module>.py tests/fixtures.py
   ```

4. **Verify formatting**: Ensure code follows project standards
   ```bash
   make lint  # or python -m black --check tests/
   ```

### Test Categories

**Unit Tests (`test_train.py`):**
- Comprehensive mocking of external dependencies
- Fast execution with isolated testing
- Tests individual function behavior

**Integration Tests (`test_sweep_integration.py`):**
- Uses real SLEAP experiment data from `tests/data/`
- Two classes: `TestSweepIntegrationWithMocks` and `TestPureIntegration`
- Tests actual workflow with minimal or no mocking
- Verifies cross-platform compatibility and path handling

### CI/CD Integration

Multiple GitHub Actions workflows run automatically:

**Comprehensive (runs on changes to main):**
- **`test-imports.yml`** - Cross-platform import validation:
  - Tests on Ubuntu, Windows, macOS
  - Python 3.8 compatibility
  - Lightweight without full SLEAP installation

**Full Integration (runs on changes to main):**
- **`ci.yml`** - Complete CI pipeline:
  - Full SLEAP installation via pip
  - Comprehensive test suite
  - Code coverage reporting
  - Package building verification

**Workflow Priority:**
1. **`test-imports.yml`** - Should pass (compatibility)
2. **`ci.yml`** - Nice to pass (full validation)

## Data Management

- Training data is stored in sleap packages with embedded images
- Labels are stored as SLEAP files (`.slp`)
- Models are stored in timestamped directories under `models/`
- All artifacts are tracked in W&B with comprehensive metadata

## Sweep Metrics Evaluation

### Quick Start - Get All Metrics from Recent Sweeps

For most use cases, this is what you need:

```python
from sleap_roots_training.evaluate import fetch_metrics_from_sweep_pattern

# Define your target metrics
TARGET_METRICS = ["dist_p50", "dist_p90", "dist_p95", "dist_avg", "vis_prec", "vis_recall"]

# Get all metrics from sweeps matching your experiment pattern
sweep_df = fetch_metrics_from_sweep_pattern(
    name_pattern="20250717_plate_medicago_primary_sweep_receptive_field",
    target_metrics=TARGET_METRICS,
    earliest_time="2025-07-17T00:00:00Z",
    include_config=True
)

print(f"Found {len(sweep_df)} runs from {sweep_df['sweep_id'].nunique()} sweeps")
print(f"Columns: {list(sweep_df.columns)}")

# Analyze results
summary = sweep_df.groupby('sweep_name')[TARGET_METRICS].agg(['mean', 'std', 'count'])
print(summary)
```

### Advanced Usage

**Group runs for future organization:**
```python
# This will also retroactively group runs with proper names
sweep_df = fetch_metrics_from_sweep_pattern(
    name_pattern="medicago_primary_sweep",
    target_metrics=TARGET_METRICS,
    group_runs=True,  # Automatically group runs
    group_name_base="medicago_receptive_field"
)
```

**Find recent experiments by prefix:**
```python
from sleap_roots_training.evaluate import find_and_evaluate_recent_sweeps

# Get all medicago experiments from the last 7 days
df = find_and_evaluate_recent_sweeps(
    experiment_prefix="medicago",
    days_back=7
)
```

**Retroactively group existing ungrouped runs:**
```python
from sleap_roots_training.evaluate import group_sweep_runs_retroactively

# Group all runs from a specific sweep ID
updated_runs = group_sweep_runs_retroactively(
    sweep_id="4zkofrue",
    group_name="20250717_plate_medicago_primary_sweep_receptive_field"
)
print(f"Updated {len(updated_runs)} runs")
```

### Key Benefits

1. **No manual sweep ID tracking** - Finds sweeps automatically by name pattern
2. **Multi-sweep support** - Handles multiple train/test splits in one dataframe
3. **Automatic grouping** - Can organize runs retroactively or during fetch
4. **Cross-platform compatibility** - Works on Windows, macOS, and Linux
5. **Integration ready** - Works with existing evaluation and visualization functions

### Migration from Old Workflow

**Before (manual sweep IDs):**
```python
sweep_ids = ["4zkofrue", "abc123", "xyz789"]  # Manual tracking
sweep_df = fetch_sweep_metrics(sweep_ids=sweep_ids, ...)
```

**After (automatic discovery):**
```python
sweep_df = fetch_metrics_from_sweep_pattern(
    name_pattern="your_experiment_name",
    target_metrics=TARGET_METRICS,
    earliest_time="2025-07-17T00:00:00Z"
)
```

This automatically finds all matching sweeps and combines metrics from all runs across all train/test splits.

## Important Notes

- Always run notebooks from repository root for proper imports
- Use separate branches for different experiments
- Model evaluation uses 17.0 px/mm as default scaling factor
- W&B runs are automatically tagged and grouped by experiment names
- Configuration files are timestamped to maintain experiment reproducibility