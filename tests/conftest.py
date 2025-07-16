"""Pytest configuration and fixtures for sleap-roots-training tests."""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch
import pandas as pd
import numpy as np


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture
def mock_config():
    """Mock configuration dictionary."""
    return {
        "project_name": "test_project",
        "entity_name": "test_entity", 
        "experiment_name": "test_experiment",
        "registry": "test_registry",
        "collection_name": "test_collection",
        "job_type": "test_job"
    }


@pytest.fixture
def mock_wandb_run():
    """Mock W&B run object."""
    mock_run = Mock()
    mock_run.id = "test_run_id"
    mock_run.name = "test_run_name"
    mock_run.group = "test_group"
    mock_run.config = {}
    mock_run.summary = {}
    return mock_run


@pytest.fixture
def mock_wandb_artifact():
    """Mock W&B artifact object."""
    mock_artifact = Mock()
    mock_artifact.name = "test_artifact"
    mock_artifact.type = "test_type"
    mock_artifact.metadata = {}
    mock_artifact.download.return_value = "/path/to/artifact"
    return mock_artifact


@pytest.fixture
def sample_training_data():
    """Sample training data DataFrame."""
    return pd.DataFrame({
        "version": [1, 1, 2, 2, 3],
        "path": [
            "/path/to/v1/config1.json",
            "/path/to/v1/config2.json", 
            "/path/to/v2/config1.json",
            "/path/to/v2/config2.json",
            "/path/to/v3/config1.json"
        ]
    })


@pytest.fixture
def sample_metrics():
    """Sample metrics dictionary for model evaluation."""
    return {
        "dist.p50": 85.0,
        "dist.p90": 170.0,
        "dist.p95": 255.0,
        "dist.p99": 340.0,
        "dist.avg": 100.0,
        "dist.dists": np.array([[85.0, 170.0], [255.0, 340.0]]),
        "vis.precision": 0.95,
        "vis.recall": 0.90,
        "oks_voc.mAP": 0.85,
        "oks_voc.mAR": 0.80
    }


@pytest.fixture
def sample_sweep_config():
    """Sample sweep configuration for testing."""
    return {
        "method": "grid",
        "name": "test_sweep",
        "parameters": {
            "learning_rate": {"values": [0.001, 0.01, 0.1]},
            "batch_size": {"values": [16, 32, 64]},
            "data.preprocessing.input_scaling": {"values": [0.5, 1.0, 1.5]}
        }
    }


@pytest.fixture
def mock_sleap_metrics():
    """Mock SLEAP metrics loading."""
    with patch('sleap_roots_training.train.sleap.load_metrics') as mock_load:
        mock_load.return_value = {
            "dist.p50": 85.0,
            "dist.p90": 170.0,
            "dist.p95": 255.0,
            "dist.p99": 340.0,
            "dist.avg": 100.0,
            "dist.dists": np.array([[85.0, 170.0], [255.0, 340.0]]),
            "vis.precision": 0.95,
            "vis.recall": 0.90,
            "oks_voc.mAP": 0.85,
            "oks_voc.mAR": 0.80
        }
        yield mock_load


@pytest.fixture
def mock_sleap_model():
    """Mock SLEAP model loading."""
    with patch('sleap_roots_training.evaluate.sleap.load_model') as mock_load:
        mock_predictor = Mock()
        mock_predictor.bottomup_model = Mock()
        mock_predictor.bottomup_config = Mock()
        mock_predictor.predict.return_value = Mock()
        mock_load.return_value = mock_predictor
        yield mock_predictor


@pytest.fixture
def mock_plt():
    """Mock matplotlib.pyplot for visualization tests."""
    with patch('sleap_roots_training.train.plt') as mock_plt:
        mock_plt.figure.return_value = Mock()
        mock_plt.savefig.return_value = None
        mock_plt.close.return_value = None
        yield mock_plt


@pytest.fixture
def mock_subprocess():
    """Mock subprocess for training execution tests."""
    with patch('sleap_roots_training.train.subprocess.run') as mock_run:
        mock_result = Mock()
        mock_result.stdout = "Training completed successfully"
        mock_result.stderr = ""
        mock_run.return_value = mock_result
        yield mock_run


@pytest.fixture
def created_config_files(temp_dir):
    """Create test configuration files in temporary directory."""
    config_files = []
    
    for version in [1, 2, 3]:
        config_dir = temp_dir / f"v{version}"
        config_dir.mkdir()
        
        config_path = config_dir / f"initial_config_modified_v00{version}.json"
        config_content = {
            "data": {
                "preprocessing": {"input_scaling": 1.0},
                "labels": {"test_labels": f"/path/to/test_v{version}.slp"}
            },
            "model": {"backbone": {"type": "resnet18"}},
            "training": {"batch_size": 16, "epochs": 100}
        }
        
        with open(config_path, 'w') as f:
            import json
            json.dump(config_content, f)
        
        config_files.append(config_path)
    
    return config_files


@pytest.fixture
def mock_logging():
    """Mock logging for testing log messages."""
    with patch('sleap_roots_training.train.logging') as mock_log:
        yield mock_log


@pytest.fixture 
def mock_wandb_api():
    """Mock W&B API for testing sweep metrics."""
    with patch('sleap_roots_training.evaluate.wandb.Api') as mock_api_class:
        mock_api = Mock()
        mock_api_class.return_value = mock_api
        
        # Mock sweep
        mock_sweep = Mock()
        mock_api.sweep.return_value = mock_sweep
        
        # Mock runs
        mock_run1 = Mock()
        mock_run1.state = "finished"
        mock_run1.id = "run1"
        mock_run1.name = "test_run_1"
        mock_run1.group = "test_group"
        mock_run1.summary = {"dist_avg": 5.0, "vis_precision": 0.9}
        mock_run1.config = {"param1": 1.0}
        mock_run1.sweep = Mock()
        mock_run1.sweep.id = "sweep1"
        
        mock_run2 = Mock()
        mock_run2.state = "finished"
        mock_run2.id = "run2"
        mock_run2.name = "test_run_2"
        mock_run2.group = "test_group"
        mock_run2.summary = {"dist_avg": 6.0, "vis_precision": 0.8}
        mock_run2.config = {"param1": 2.0}
        mock_run2.sweep = Mock()
        mock_run2.sweep.id = "sweep2"
        
        mock_sweep.runs = [mock_run1, mock_run2]
        mock_api.runs.return_value = [mock_run1, mock_run2]
        
        yield mock_api


@pytest.fixture(autouse=True)
def reset_config_module():
    """Reset config module state between tests."""
    # This ensures that CONFIG changes in one test don't affect others
    import sleap_roots_training.config as config_module
    
    original_config = config_module.CONFIG.copy()
    yield
    
    # Restore original config
    config_module.CONFIG.clear()
    config_module.CONFIG.update(original_config)


# Mark slow tests - will be configured in pytest_configure
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )


def pytest_addoption(parser):
    """Add command line options for pytest."""
    parser.addoption(
        "--no-slow", 
        action="store_true", 
        default=False,
        help="Skip slow tests"
    )
    parser.addoption(
        "--integration",
        action="store_true",
        default=False,
        help="Run integration tests"
    )