"""
Shared pytest fixtures for sleap-roots-training tests.

This module contains reusable fixtures that can be imported and used across
multiple test modules to ensure consistent test data and setup.
"""

import pytest
import tempfile
import json
import pandas as pd
from pathlib import Path
import shutil
import numpy as np

from sleap_roots_training.train import load_training_data


@pytest.fixture
def sweep_experiment_data():
    """
    Fixture providing sweep experiment data for integration tests.

    This fixture loads the real CSV and JSON config files, but creates mock SLEAP
    package files to avoid large file size issues in GitHub.

    Returns:
        dict: Dictionary containing paths and data for the sweep experiment:
            - csv_path: Path to the train_test_splits.csv file
            - config_path: Path to the initial configuration JSON file
            - config: Loaded configuration dictionary
            - df: Pandas DataFrame with the CSV data
            - data_dir: Path to the data directory
            - test_data_dir: Path to the root test data directory
    """
    # Path to test data directory
    test_data_dir = Path(__file__).parent / "data" / "sweep_experiment"

    # Load the CSV data
    csv_path = test_data_dir / "train_test_splits.csv"
    df = load_training_data(csv_path)

    # Get the config path
    config_path = (
        test_data_dir / "train_test_split.v000" / "initial_config_modified_v000.json"
    )

    # Create the directory if it doesn't exist
    config_path.parent.mkdir(parents=True, exist_ok=True)

    # Load the real configuration file
    with open(config_path, "r") as f:
        config = json.load(f)

    # Update paths to use local test data with proper path separators
    local_data_dir = test_data_dir / "train_test_split.v000"
    local_data_dir.mkdir(parents=True, exist_ok=True)

    # Create mock SLEAP files if they don't exist (to avoid large file issues)
    slp_files = ["train.pkg.slp", "val.pkg.slp", "test.pkg.slp"]
    for slp_file in slp_files:
        slp_path = local_data_dir / slp_file
        if not slp_path.exists():
            # Create a small mock SLEAP file
            slp_path.write_text(f"Mock SLEAP file: {slp_file}")

    config["data"]["labels"]["training_labels"] = str(local_data_dir / "train.pkg.slp")
    config["data"]["labels"]["validation_labels"] = str(local_data_dir / "val.pkg.slp")
    config["data"]["labels"]["test_labels"] = str(local_data_dir / "test.pkg.slp")

    return {
        "csv_path": csv_path,
        "config_path": config_path,
        "config": config,
        "df": df,
        "data_dir": local_data_dir,
        "test_data_dir": test_data_dir,
    }


@pytest.fixture
def temp_experiment_dir(sweep_experiment_data):
    """
    Create a temporary directory with copied experiment data for safe testing.

    This fixture creates a temporary directory and copies the sweep experiment data
    to it, ensuring that tests don't modify the original test data. All paths are
    updated to use forward slashes for cross-platform compatibility.

    Args:
        sweep_experiment_data: The base sweep experiment data fixture

    Yields:
        dict: Dictionary containing paths to the temporary experiment data:
            - experiment_dir: Path to the temporary experiment directory
            - csv_path: Path to the CSV file in temp directory
            - config_path: Path to the config file in temp directory
            - config: Updated configuration dictionary
            - data_dir: Path to the data directory in temp directory
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Copy the experiment data to temp directory
        experiment_dir = temp_path / "sweep_experiment"
        shutil.copytree(sweep_experiment_data["test_data_dir"], experiment_dir)

        # Update the CSV to point to the temporary directory
        csv_path = experiment_dir / "train_test_splits.csv"
        df = pd.read_csv(csv_path)

        # Update paths in CSV to point to temp directory with forward slashes
        # Replace Windows network paths with local temp paths
        for idx, row in df.iterrows():
            old_path = row["path"]
            # Extract just the filename (e.g., "train.pkg.slp") from the network path
            if "\\" in old_path:
                # Handle Windows UNC paths
                filename = old_path.split("\\")[-1]
            else:
                # Handle regular paths
                filename = Path(old_path).name
            # Create new path in temp directory
            new_path = experiment_dir / "train_test_split.v000" / filename
            # Convert to string with forward slashes for cross-platform compatibility
            df.at[idx, "path"] = str(new_path).replace("\\", "/")

        df.to_csv(csv_path, index=False)

        # Update config file paths
        config_path = (
            experiment_dir
            / "train_test_split.v000"
            / "initial_config_modified_v000.json"
        )

        # Load config if it exists, otherwise use the config from sweep_experiment_data
        if config_path.exists():
            with open(config_path, "r") as f:
                config = json.load(f)
        else:
            # Use the config from sweep_experiment_data and create the file
            config = sweep_experiment_data["config"].copy()
            with open(config_path, "w") as f:
                json.dump(config, f, indent=2)

        local_data_dir = experiment_dir / "train_test_split.v000"
        # Use forward slashes for cross-platform compatibility
        config["data"]["labels"]["training_labels"] = str(
            local_data_dir / "train.pkg.slp"
        ).replace("\\", "/")
        config["data"]["labels"]["validation_labels"] = str(
            local_data_dir / "val.pkg.slp"
        ).replace("\\", "/")
        config["data"]["labels"]["test_labels"] = str(
            local_data_dir / "test.pkg.slp"
        ).replace("\\", "/")
        config["outputs"]["runs_folder"] = str(local_data_dir / "models").replace(
            "\\", "/"
        )

        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)

        yield {
            "experiment_dir": experiment_dir,
            "csv_path": csv_path,
            "config_path": config_path,
            "config": config,
            "data_dir": local_data_dir,
        }


@pytest.fixture
def realistic_sweep_config():
    """
    Fixture providing a realistic sweep configuration for medicago experiments.

    This fixture returns a W&B sweep configuration that matches the parameters
    typically used in SLEAP roots training experiments.

    Returns:
        dict: W&B sweep configuration dictionary with realistic parameters
    """
    return {
        "method": "grid",
        "parameters": {
            "data.preprocessing.input_scaling": {"values": [0.5, 1.0, 1.5]},
            "model.backbone.unet.max_stride": {"values": [16, 32]},
            "model.backbone.unet.filters": {"values": [24, 32]},
            "optimization.batch_size": {"values": [2, 4]},
            "optimization.initial_learning_rate": {"values": [0.0001, 0.001]},
        },
    }


@pytest.fixture
def small_sweep_config():
    """
    Fixture providing a small sweep configuration for faster testing.

    This fixture returns a minimal W&B sweep configuration with fewer parameters
    for tests that need to run quickly.

    Returns:
        dict: W&B sweep configuration dictionary with minimal parameters
    """
    return {
        "method": "grid",
        "parameters": {
            "data.preprocessing.input_scaling": {"values": [0.5, 1.0]},
            "optimization.batch_size": {"values": [2, 4]},
        },
    }


@pytest.fixture
def mock_models_dir(tmp_path):
    """
    Fixture providing a mock models directory structure for testing.

    This fixture creates a temporary directory structure that mimics the
    output of SLEAP training runs for testing model discovery and evaluation.

    Args:
        tmp_path: pytest's tmp_path fixture

    Returns:
        Path: Path to the mock models directory
    """
    models_dir = tmp_path / "models"
    models_dir.mkdir()

    # Create mock run directories with timestamps
    run_dirs = [
        "run_241201_120000",  # oldest
        "run_241201_140000",  # newest
        "run_241201_100000",  # middle
    ]

    for run_dir in run_dirs:
        run_path = models_dir / run_dir
        run_path.mkdir()

        # Create mock training files
        (run_path / "training_config.json").write_text('{"test": "config"}')
        (run_path / "model.h5").write_text("mock model")
        (run_path / "metrics.json").write_text('{"test": "metrics"}')

    return models_dir


@pytest.fixture
def environment_config():
    """
    Fixture providing environment configuration for testing.

    This fixture returns configuration values that can be used to set up
    test environments consistently.

    Returns:
        dict: Environment configuration dictionary
    """
    return {
        "project_name": "sleap-roots-test",
        "entity_name": "test-entity",
        "experiment_name": "test-experiment",
        "registry_name": "test-registry",
        "tags": ["test", "integration"],
        "model_tags": ["test-model", "integration"],
    }


def _build_tiny_labels(img_dir):
    """Build a 3-frame, 2-node SLEAP Labels backed by real PNG files.

    Uses image files (not numpy) so the embedded package's source video is a lazy
    SingleImageVideo — a numpy source makes HDF5Video.has_embedded_images raise on load.
    """
    import os
    import imageio.v2 as imageio
    import sleap

    os.makedirs(img_dir, exist_ok=True)
    paths = []
    for i in range(3):
        p = os.path.join(img_dir, f"frame_{i}.png")
        imageio.imwrite(p, np.random.randint(0, 255, size=(16, 16), dtype=np.uint8))
        paths.append(p)

    video = sleap.Video.from_image_filenames(paths)
    skeleton = sleap.Skeleton()
    skeleton.add_node("a")
    skeleton.add_node("b")
    frames = [
        sleap.LabeledFrame(
            video=video,
            frame_idx=i,
            instances=[
                sleap.Instance.from_pointsarray(
                    np.array([[1 + i, 1 + i], [2 + i, 2 + i]]), skeleton=skeleton
                )
            ],
        )
        for i in range(2)
    ]
    return sleap.Labels(frames), paths


@pytest.fixture
def embedded_package(tmp_path):
    """Path to a tiny .pkg.slp saved WITH embedded images."""
    pytest.importorskip("sleap")
    pytest.importorskip("imageio")
    labels, _ = _build_tiny_labels(str(tmp_path / "imgs"))
    out = str(tmp_path / "embedded.pkg.slp")
    labels.save(out, with_images=True)
    return out


@pytest.fixture
def nonembedded_package(tmp_path):
    """Path to a tiny .slp saved WITHOUT embedded images (references PNGs that exist)."""
    pytest.importorskip("sleap")
    pytest.importorskip("imageio")
    labels, _ = _build_tiny_labels(str(tmp_path / "imgs"))
    out = str(tmp_path / "nonembedded.slp")
    labels.save(out, with_images=False)
    return out
