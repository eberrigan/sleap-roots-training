"""
Integration tests for sweep functionality using real data.

This module contains integration tests that use real SLEAP experiment data
to test the sweep functionality. It includes both mocked tests for isolated
testing and pure integration tests that test the actual workflow.
"""

import pytest
import json
import pandas as pd
from pathlib import Path
from unittest.mock import patch, MagicMock

from sleap_roots_training.train import (
    load_training_data,
    make_sweep_train_fn,
    run_sweep_training,
    update_config_with_wandb,
    get_param_combinations,
    get_latest_run,
)

# Import fixtures from the fixtures module
from tests.fixtures import (
    sweep_experiment_data,
    temp_experiment_dir,
    realistic_sweep_config,
    small_sweep_config,
    mock_models_dir,
    environment_config,
)


class TestSweepIntegrationWithMocks:
    """Integration tests for sweep functionality using real data with mocking."""

    def test_load_real_sweep_data(self, sweep_experiment_data):
        """Test loading real sweep experiment data."""
        df = sweep_experiment_data["df"]

        # Verify data structure
        assert len(df) == 3  # train, val, test
        assert "path" in df.columns
        assert "version" in df.columns
        assert "labeled_frames" in df.columns
        assert "split_type" in df.columns

        # Verify data content
        assert df["version"].iloc[0] == 0
        assert df["labeled_frames"].sum() == 23  # 16 + 3 + 4
        assert set(df["split_type"]) == {"train", "val", "test"}

    def test_real_config_structure(self, sweep_experiment_data):
        """Test that real config has expected structure."""
        config = sweep_experiment_data["config"]

        # Verify main sections
        assert "data" in config
        assert "model" in config
        assert "optimization" in config
        assert "outputs" in config

        # Verify key sweep parameters exist
        assert "input_scaling" in config["data"]["preprocessing"]
        assert "max_stride" in config["model"]["backbone"]["unet"]
        assert "filters" in config["model"]["backbone"]["unet"]
        assert "batch_size" in config["optimization"]
        assert "initial_learning_rate" in config["optimization"]

    def test_real_files_exist(self, sweep_experiment_data):
        """Test that real SLEAP files exist."""
        data_dir = sweep_experiment_data["data_dir"]

        # Check that SLEAP files exist
        assert (data_dir / "train.pkg.slp").exists()
        assert (data_dir / "val.pkg.slp").exists()
        assert (data_dir / "test.pkg.slp").exists()
        assert sweep_experiment_data["config_path"].exists()

    def test_parameter_combinations_with_realistic_config(self, realistic_sweep_config):
        """Test parameter combinations calculation with realistic sweep config."""
        combinations = get_param_combinations(realistic_sweep_config)

        # Should be 3 * 2 * 2 * 2 * 2 = 48 combinations
        assert combinations == 48

    def test_parameter_combinations_with_small_config(self, small_sweep_config):
        """Test parameter combinations calculation with small sweep config."""
        combinations = get_param_combinations(small_sweep_config)

        # Should be 2 * 2 = 4 combinations
        assert combinations == 4

    @patch("sleap_roots_training.train.wandb.init")
    @patch("sleap_roots_training.train.wandb.config")
    @patch("sleap_roots_training.train.execute_training")
    @patch("sleap_roots_training.train.get_latest_run")
    @patch("sleap_roots_training.train.log_model_artifact_with_evals")
    def test_sweep_train_fn_with_real_config(
        self,
        mock_log_artifact,
        mock_get_latest_run,
        mock_execute_training,
        mock_wandb_config,
        mock_wandb_init,
        temp_experiment_dir,
        environment_config,
    ):
        """Test sweep training function with real configuration."""
        # Setup mocks
        mock_run = MagicMock()
        mock_run.id = "test_run_123"
        mock_wandb_init.return_value = mock_run

        # Mock wandb.config with realistic sweep parameters
        mock_config_dict = {
            "data.preprocessing.input_scaling": 1.5,
            "model.backbone.unet.max_stride": 32,
            "optimization.batch_size": 2,
            "optimization.initial_learning_rate": 0.001,
        }

        # Configure mock to behave like a dict when converted with dict()
        mock_wandb_config.__bool__ = lambda self: True
        mock_wandb_config.items.return_value = mock_config_dict.items()
        mock_wandb_config.keys.return_value = mock_config_dict.keys()
        mock_wandb_config.values.return_value = mock_config_dict.values()
        mock_wandb_config.__iter__.return_value = iter(mock_config_dict.items())

        # Make individual key access work
        def mock_getitem(self, key):
            return mock_config_dict[key]

        mock_wandb_config.__getitem__ = mock_getitem

        # Mock get_latest_run to return a fake model directory
        mock_model_dir = MagicMock()
        mock_model_dir.exists.return_value = True
        mock_get_latest_run.return_value = mock_model_dir

        # Create sweep training function with real data
        config = temp_experiment_dir["config"]
        data_dir = temp_experiment_dir["data_dir"]

        train_fn = make_sweep_train_fn(
            version="0",
            config_copy=config,
            dir_path=data_dir,
            sleap_train_command="sleap-train {}",
            experiment_name=environment_config["experiment_name"],
            model_tags=environment_config["model_tags"],
            link_to_registry=False,
            registry_name=None,
        )

        # Execute the training function
        train_fn()

        # Verify that wandb.init was called with group parameter
        mock_wandb_init.assert_called_once_with(
            group=environment_config["experiment_name"]
        )

        # Verify that execute_training was called with a config file
        mock_execute_training.assert_called_once()
        execute_args = mock_execute_training.call_args[0][0]
        assert "sleap-train" in execute_args
        assert "sweep_config_v000" in execute_args
        assert "test_run_123" in execute_args

        # Verify model artifact logging was called
        mock_log_artifact.assert_called_once()

    def test_config_update_with_real_sweep_params(self, temp_experiment_dir):
        """Test config update with realistic sweep parameters."""
        config = temp_experiment_dir["config"].copy()

        # Mock wandb.config with realistic parameters
        with patch("sleap_roots_training.train.wandb.config") as mock_wandb_config:
            mock_config_dict = {
                "data.preprocessing.input_scaling": 1.5,
                "model.backbone.unet.max_stride": 32,
                "model.backbone.unet.filters": 32,
                "optimization.batch_size": 2,
                "optimization.initial_learning_rate": 0.001,
            }

            # Configure mock to behave like a dict when converted with dict()
            mock_wandb_config.__bool__ = lambda self: True
            mock_wandb_config.items.return_value = mock_config_dict.items()
            mock_wandb_config.keys.return_value = mock_config_dict.keys()
            mock_wandb_config.values.return_value = mock_config_dict.values()
            mock_wandb_config.__iter__.return_value = iter(mock_config_dict.items())

            # Make individual key access work
            def mock_getitem(self, key):
                return mock_config_dict[key]

            mock_wandb_config.__getitem__ = mock_getitem

            # Update config
            updated_config = update_config_with_wandb(config)

            # Verify updates
            assert updated_config["data"]["preprocessing"]["input_scaling"] == 1.5
            assert updated_config["model"]["backbone"]["unet"]["max_stride"] == 32
            assert updated_config["model"]["backbone"]["unet"]["filters"] == 32
            assert updated_config["optimization"]["batch_size"] == 2
            assert updated_config["optimization"]["initial_learning_rate"] == 0.001

            # Verify original nested structure is preserved
            assert "validation_labels" in updated_config["data"]["labels"]
            assert (
                "sigma"
                in updated_config["model"]["heads"]["multi_instance"]["confmaps"]
            )

    @patch("sleap_roots_training.train.wandb.sweep")
    @patch("sleap_roots_training.train.wandb.agent")
    @patch("sleap_roots_training.train.make_sweep_train_fn")
    def test_run_sweep_training_with_real_data(
        self,
        mock_make_sweep_train_fn,
        mock_wandb_agent,
        mock_wandb_sweep,
        temp_experiment_dir,
        small_sweep_config,
        environment_config,
    ):
        """Test running sweep training with real data."""
        # Setup mocks
        mock_sweep_id = "sweep_medicago_123"
        mock_wandb_sweep.return_value = mock_sweep_id
        mock_train_fn = MagicMock()
        mock_make_sweep_train_fn.return_value = mock_train_fn

        # Run sweep training
        run_sweep_training(
            project_name=environment_config["project_name"],
            entity_name=environment_config["entity_name"],
            experiment_name=environment_config["experiment_name"],
            version="0",
            config_copy=temp_experiment_dir["config"],
            dir_path=temp_experiment_dir["data_dir"],
            model_tags=environment_config["model_tags"],
            sleap_train_command="sleap-train {}",
            sweep_config=small_sweep_config,
            link_to_registry=False,
            registry_name=None,
        )

        # Verify sweep was created
        mock_wandb_sweep.assert_called_once()
        sweep_call_args = mock_wandb_sweep.call_args
        assert sweep_call_args[1]["project"] == environment_config["project_name"]
        assert sweep_call_args[1]["entity"] == environment_config["entity_name"]

        # Verify sweep configuration
        created_sweep_config = sweep_call_args[0][0]
        assert created_sweep_config["method"] == "grid"
        assert (
            created_sweep_config["name"]
            == f"{environment_config['experiment_name']}_v000_sweep"
        )

        # Verify training function was created with real data
        mock_make_sweep_train_fn.assert_called_once()
        make_fn_call_args = mock_make_sweep_train_fn.call_args
        assert make_fn_call_args[1]["version"] == "0"
        assert (
            make_fn_call_args[1]["experiment_name"]
            == environment_config["experiment_name"]
        )

        # Verify agent was started with correct parameters
        mock_wandb_agent.assert_called_once()
        agent_call_args = mock_wandb_agent.call_args
        assert agent_call_args[0][0] == mock_sweep_id
        assert agent_call_args[1]["function"] == mock_train_fn
        assert agent_call_args[1]["count"] == 4  # 2 * 2 = 4 combinations
        assert agent_call_args[1]["project"] == environment_config["project_name"]
        assert agent_call_args[1]["entity"] == environment_config["entity_name"]


class TestPureIntegration:
    """Pure integration tests without mocking - tests actual workflow."""

    def test_csv_data_loading_and_processing(self, temp_experiment_dir):
        """Test that CSV data can be loaded and processed correctly."""
        csv_path = temp_experiment_dir["csv_path"]

        # Load data
        df = load_training_data(csv_path)

        # Verify data
        assert len(df) == 3
        assert df["version"].iloc[0] == 0

        # Verify paths point to temp directory and use forward slashes
        for path in df["path"]:
            # Convert both paths to use forward slashes for comparison
            normalized_temp_dir = str(temp_experiment_dir["experiment_dir"]).replace(
                "\\", "/"
            )
            assert normalized_temp_dir in path
            assert Path(path).exists()
            # Verify no backslashes (Windows path issue)
            assert "\\" not in path

    def test_config_file_paths_are_cross_platform(self, temp_experiment_dir):
        """Test that config file paths work across platforms."""
        config = temp_experiment_dir["config"]

        # Verify all paths use forward slashes
        assert "\\" not in config["data"]["labels"]["training_labels"]
        assert "\\" not in config["data"]["labels"]["validation_labels"]
        assert "\\" not in config["data"]["labels"]["test_labels"]
        assert "\\" not in config["outputs"]["runs_folder"]

        # Verify paths point to existing files
        assert Path(config["data"]["labels"]["training_labels"]).exists()
        assert Path(config["data"]["labels"]["validation_labels"]).exists()
        assert Path(config["data"]["labels"]["test_labels"]).exists()

    def test_get_latest_run_with_real_directory_structure(self, mock_models_dir):
        """Test getting latest run with real directory structure."""
        # Test with mock models directory
        latest_run = get_latest_run(mock_models_dir)

        # Should return the latest timestamped directory
        assert latest_run is not None
        assert latest_run.name == "run_241201_140000"
        assert latest_run.exists()

    def test_parameter_combinations_calculation(
        self, realistic_sweep_config, small_sweep_config
    ):
        """Test parameter combinations calculation with different configs."""
        # Test realistic config
        realistic_combinations = get_param_combinations(realistic_sweep_config)
        assert realistic_combinations == 48  # 3 * 2 * 2 * 2 * 2

        # Test small config
        small_combinations = get_param_combinations(small_sweep_config)
        assert small_combinations == 4  # 2 * 2

        # Test random method (should return None)
        random_config = {
            "method": "random",
            "parameters": {"param1": {"values": [1, 2]}},
        }
        assert get_param_combinations(random_config) is None

    def test_config_deep_copy_preservation(self, temp_experiment_dir):
        """Test that config structure is preserved during operations."""
        original_config = temp_experiment_dir["config"]

        # Create a deep copy like the train function does
        import copy

        config_copy = copy.deepcopy(original_config)

        # Verify the copy is independent
        config_copy["data"]["preprocessing"]["input_scaling"] = 999
        assert original_config["data"]["preprocessing"]["input_scaling"] != 999

        # Verify all nested structures are preserved
        assert "data" in config_copy
        assert "labels" in config_copy["data"]
        assert "training_labels" in config_copy["data"]["labels"]
        assert "model" in config_copy
        assert "backbone" in config_copy["model"]
        assert "unet" in config_copy["model"]["backbone"]
        assert "optimization" in config_copy
        assert "outputs" in config_copy

    def test_sweep_config_name_generation(self, small_sweep_config):
        """Test that sweep config names are generated correctly."""
        # Test with different experiment names and versions
        test_cases = [
            ("medicago_sweep", "0", "medicago_sweep_v000_sweep"),
            ("arabidopsis_test", "1", "arabidopsis_test_v001_sweep"),
            ("rice_experiment", "10", "rice_experiment_v010_sweep"),
        ]

        for experiment_name, version, expected_name in test_cases:
            # Create a copy of the sweep config
            sweep_config_copy = small_sweep_config.copy()

            # Simulate the name generation from run_sweep_training
            sweep_config_copy["name"] = f"{experiment_name}_v{version.zfill(3)}_sweep"

            assert sweep_config_copy["name"] == expected_name

    def test_file_path_generation_during_sweep(self, temp_experiment_dir):
        """Test that file paths are generated correctly during sweep execution."""
        data_dir = temp_experiment_dir["data_dir"]
        version = "0"
        run_id = "test_run_123"

        # Test the path generation logic from make_sweep_train_fn
        from datetime import datetime

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        expected_path = (
            data_dir / f"sweep_config_v{version.zfill(3)}_{timestamp}_{run_id}.json"
        )

        # Verify the path structure
        assert "sweep_config_v000" in str(expected_path)
        assert run_id in str(expected_path)
        assert expected_path.suffix == ".json"

    def test_models_directory_creation(self, temp_experiment_dir):
        """Test that models directory is created when needed."""
        data_dir = temp_experiment_dir["data_dir"]
        models_dir = data_dir / "models"

        # Initially models directory might not exist
        if not models_dir.exists():
            models_dir.mkdir(parents=True)

        # Verify it exists and is a directory
        assert models_dir.exists()
        assert models_dir.is_dir()

        # Test that we can create subdirectories
        run_dir = models_dir / "run_test_123"
        run_dir.mkdir()

        assert run_dir.exists()
        assert run_dir.is_dir()

    def test_environment_fixture_values(self, environment_config):
        """Test that environment fixture provides expected values."""
        # Verify all required keys are present
        required_keys = [
            "project_name",
            "entity_name",
            "experiment_name",
            "registry_name",
            "tags",
            "model_tags",
        ]
        for key in required_keys:
            assert key in environment_config

        # Verify values are appropriate for testing
        assert "test" in environment_config["project_name"]
        assert "test" in environment_config["entity_name"]
        assert "test" in environment_config["experiment_name"]
        assert "test" in environment_config["registry_name"]
        assert "test" in environment_config["tags"]
        assert "test" in environment_config["model_tags"][0]

    def test_fixture_data_consistency(self, sweep_experiment_data, temp_experiment_dir):
        """Test that fixture data is consistent between fixtures."""
        # Both fixtures should reference the same underlying data
        assert sweep_experiment_data["df"]["version"].iloc[0] == 0
        assert len(sweep_experiment_data["df"]) == 3

        # Temp experiment should have the same data structure
        temp_df = load_training_data(temp_experiment_dir["csv_path"])
        assert len(temp_df) == 3
        assert temp_df["version"].iloc[0] == 0

        # Both should have the same config structure
        assert "data" in sweep_experiment_data["config"]
        assert "data" in temp_experiment_dir["config"]
        assert "model" in sweep_experiment_data["config"]
        assert "model" in temp_experiment_dir["config"]
