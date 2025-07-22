"""Tests for sleap_roots_training.train module."""

import pytest
import tempfile
import json
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock, Mock, mock_open, call
from datetime import datetime
import subprocess
import shutil

from sleap_roots_training.train import (
    load_training_data,
    get_training_groups,
    log_to_wandb,
    execute_training,
    log_model_artifact,
    evaluate_model_and_generate_visuals,
    update_config_with_wandb,
    get_latest_run,
    get_param_combinations,
    log_model_artifact_with_evals,
    run_single_training,
    make_sweep_train_fn,
    run_sweep_training,
    main,
)
from tests.fixtures import (
    temp_experiment_dir,
    mock_models_dir,
    environment_config,
    realistic_sweep_config,
    small_sweep_config,
)


class TestLoadTrainingData:
    """Test suite for load_training_data function."""

    def test_load_training_data(self):
        """Test loading training data from CSV."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("version,path\n1,/path/to/v1\n2,/path/to/v2\n")
            f.flush()
            temp_file = f.name

        df = load_training_data(temp_file)

        assert len(df) == 2
        assert list(df.columns) == ["version", "path"]
        assert df.iloc[0]["version"] == 1
        assert df.iloc[1]["path"] == "/path/to/v2"

        # Clean up
        try:
            Path(temp_file).unlink()
        except PermissionError:
            pass  # File might still be locked on Windows


class TestGetTrainingGroups:
    """Test suite for get_training_groups function."""

    def test_get_training_groups(self):
        """Test grouping training data by version."""
        df = pd.DataFrame(
            {
                "version": [1, 1, 2, 2],
                "path": ["/path/v1/a", "/path/v1/b", "/path/v2/a", "/path/v2/b"],
            }
        )

        groups = get_training_groups(df)

        assert len(groups) == 2
        group_dict = {name: group for name, group in groups}
        assert 1 in group_dict
        assert 2 in group_dict
        assert len(group_dict[1]) == 2
        assert len(group_dict[2]) == 2


class TestLogToWandb:
    """Test suite for log_to_wandb function."""

    @patch("sleap_roots_training.train.wandb.config")
    @patch("sleap_roots_training.train.wandb.init")
    def test_log_to_wandb_basic(self, mock_wandb_init, mock_wandb_config):
        """Test basic W&B logging."""
        mock_run = MagicMock()
        mock_wandb_init.return_value = mock_run

        config = {"test": "config"}
        config_path = Path("/test/config.json")

        result = log_to_wandb(
            project_name="test_project",
            entity_name="test_entity",
            experiment_name="test_experiment",
            version="1",
            config=config,
            config_path=config_path,
            tags=["test"],
        )

        mock_wandb_init.assert_called_once_with(
            project="test_project",
            entity="test_entity",
            group="test_experiment",
            config=config,
            name="test_experiment_training_v001",
            tags=["test"],
            mode="online",
        )

        mock_wandb_config.update.assert_called_once_with(
            {"version": "1", "config_path": config_path.as_posix()}
        )

        assert result == mock_run

    @patch("sleap_roots_training.train.wandb.config")
    @patch("sleap_roots_training.train.wandb.init")
    def test_log_to_wandb_no_tags(self, mock_wandb_init, mock_wandb_config):
        """Test W&B logging without tags."""
        mock_run = MagicMock()
        mock_wandb_init.return_value = mock_run

        result = log_to_wandb(
            project_name="test_project",
            entity_name="test_entity",
            experiment_name="test_experiment",
            version="2",
            config={},
            config_path=Path("/test/config.json"),
        )

        mock_wandb_init.assert_called_once()
        args, kwargs = mock_wandb_init.call_args
        assert kwargs["tags"] is None
        assert kwargs["name"] == "test_experiment_training_v002"


class TestExecuteTraining:
    """Test suite for execute_training function."""

    @patch("sleap_roots_training.train.subprocess.run")
    @patch("builtins.print")
    def test_execute_training_success(self, mock_print, mock_subprocess_run):
        """Test successful training execution."""
        mock_result = MagicMock()
        mock_result.stdout = "Training completed successfully"
        mock_subprocess_run.return_value = mock_result

        command = "sleap-train config.json"
        execute_training(command)

        import subprocess

        mock_subprocess_run.assert_called_once_with(
            command,
            shell=True,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        mock_print.assert_any_call(f"Executing: {command}")
        mock_print.assert_any_call("Training completed successfully")

    @patch("sleap_roots_training.train.subprocess.run")
    @patch("builtins.print")
    def test_execute_training_failure(self, mock_print, mock_subprocess_run):
        """Test training execution failure."""
        from subprocess import CalledProcessError

        mock_error = CalledProcessError(1, "sleap-train")
        mock_error.stderr = "Training failed with error"
        mock_subprocess_run.side_effect = mock_error

        command = "sleap-train config.json"

        with pytest.raises(CalledProcessError):
            execute_training(command)

        mock_print.assert_any_call(f"Executing: {command}")
        mock_print.assert_any_call(
            "Error executing training command: Training failed with error"
        )


class TestLogModelArtifact:
    """Test suite for log_model_artifact function."""

    @patch("sleap_roots_training.train.wandb.Artifact")
    @patch("builtins.print")
    def test_log_model_artifact_basic(self, mock_print, mock_artifact_class):
        """Test basic model artifact logging."""
        mock_run = MagicMock()
        mock_artifact = MagicMock()
        mock_artifact.metadata = {}
        mock_artifact_class.return_value = mock_artifact

        with tempfile.TemporaryDirectory() as temp_dir:
            model_dir = Path(temp_dir)

            # Create training config file
            config_path = model_dir / "training_config.json"
            test_config = {"test": "config"}
            with open(config_path, "w") as f:
                json.dump(test_config, f)

            log_model_artifact(
                run=mock_run,
                experiment_name="test_experiment",
                model_tags=["test", "model"],
                model_dir=model_dir,
                version="1",
            )

            mock_artifact_class.assert_called_once_with(
                name="test_experiment_v001",
                type="model",
                metadata={
                    "experiment": "test_experiment",
                    "version": "1",
                    **test_config,
                },
            )

            mock_artifact.add_dir.assert_called_once_with(model_dir)
            mock_run.log_artifact.assert_called_once_with(
                mock_artifact, type="model", tags=["test", "model"]
            )
            mock_run.config.update.assert_called_once_with(test_config)

    @patch("sleap_roots_training.train.wandb.Artifact")
    @patch("builtins.print")
    def test_log_model_artifact_no_config(self, mock_print, mock_artifact_class):
        """Test model artifact logging without training config."""
        mock_run = MagicMock()
        mock_artifact = MagicMock()
        mock_artifact.metadata = {}
        mock_artifact_class.return_value = mock_artifact

        with tempfile.TemporaryDirectory() as temp_dir:
            model_dir = Path(temp_dir)

            log_model_artifact(
                run=mock_run,
                experiment_name="test_experiment",
                model_tags=["test"],
                model_dir=model_dir,
                version="1",
            )

            mock_artifact_class.assert_called_once_with(
                name="test_experiment_v001",
                type="model",
                metadata={"experiment": "test_experiment", "version": "1"},
            )

            # Config update should not be called
            mock_run.config.update.assert_not_called()


class TestEvaluateModelAndGenerateVisuals:
    """Test suite for evaluate_model_and_generate_visuals function."""

    @patch("sleap_roots_training.train.sleap.load_metrics")
    @patch("sleap_roots_training.train.plt.savefig")
    @patch("sleap_roots_training.train.plt.close")
    @patch("sleap_roots_training.train.sns.histplot")
    @patch("sleap_roots_training.train.plt.figure")
    def test_evaluate_model_and_generate_visuals(
        self, mock_figure, mock_histplot, mock_close, mock_savefig, mock_load_metrics
    ):
        """Test model evaluation and visualization generation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            model_dir = Path(temp_dir)

            # Mock metrics
            mock_metrics = {
                "dist.p50": 85.0,
                "dist.p90": 170.0,
                "dist.p95": 255.0,
                "dist.p99": 340.0,
                "dist.avg": 100.0,
                "dist.dists": np.array([[85.0, 170.0], [255.0, 340.0]]),
                "vis.precision": 0.95,
                "vis.recall": 0.90,
                "oks_voc.mAP": 0.85,
                "oks_voc.mAR": 0.80,
            }
            mock_load_metrics.return_value = mock_metrics

            metrics_df, dists_df, visualizations = evaluate_model_and_generate_visuals(
                model_dir=model_dir, px_per_mm=17.0
            )

            # Check metrics DataFrame
            assert len(metrics_df) == 1
            assert metrics_df.iloc[0]["dist_p50"] == 85.0 / 17.0
            assert metrics_df.iloc[0]["dist_avg"] == 100.0 / 17.0
            assert metrics_df.iloc[0]["vis_prec"] == 0.95
            assert metrics_df.iloc[0]["oks_map"] == 0.85

            # Check distances DataFrame
            assert len(dists_df) == 4  # Flattened array
            assert "distances_mm" in dists_df.columns

            # Check visualizations
            assert "distance_histogram" in visualizations
            assert (
                visualizations["distance_histogram"]
                == model_dir / "distance_histogram.png"
            )

            mock_load_metrics.assert_called_once_with(
                model_dir.as_posix(), split="test"
            )
            mock_savefig.assert_called_once()
            mock_close.assert_called_once()

    def test_evaluate_model_nonexistent_directory(self):
        """Test evaluation with nonexistent model directory."""
        with pytest.raises(FileNotFoundError, match="Model directory not found"):
            evaluate_model_and_generate_visuals("/nonexistent/directory")

    @patch("sleap_roots_training.train.sleap.load_metrics")
    @patch("sleap_roots_training.train.plt.savefig")
    @patch("sleap_roots_training.train.plt.close")
    @patch("sleap_roots_training.train.sns.histplot")
    @patch("sleap_roots_training.train.plt.figure")
    def test_evaluate_model_px_per_mm_none(
        self, mock_figure, mock_histplot, mock_close, mock_savefig, mock_load_metrics
    ):
        """Test model evaluation with px_per_mm=None (no conversion)."""
        with tempfile.TemporaryDirectory() as temp_dir:
            model_dir = Path(temp_dir)

            # Mock metrics
            mock_metrics = {
                "dist.p50": 85.0,
                "dist.p90": 170.0,
                "dist.p95": 255.0,
                "dist.p99": 340.0,
                "dist.avg": 100.0,
                "dist.dists": np.array([[85.0, 170.0], [255.0, 340.0]]),
                "vis.precision": 0.95,
                "vis.recall": 0.90,
                "oks_voc.mAP": 0.85,
                "oks_voc.mAR": 0.80,
            }
            mock_load_metrics.return_value = mock_metrics

            metrics_df, dists_df, visualizations = evaluate_model_and_generate_visuals(
                model_dir=model_dir, px_per_mm=None
            )

            # Check metrics DataFrame - values should NOT be converted (remain in pixels)
            assert len(metrics_df) == 1
            assert metrics_df.iloc[0]["dist_p50"] == 85.0  # No conversion
            assert metrics_df.iloc[0]["dist_avg"] == 100.0  # No conversion
            assert metrics_df.iloc[0]["vis_prec"] == 0.95
            assert metrics_df.iloc[0]["oks_map"] == 0.85

            # Check distances DataFrame - should be in pixels
            assert len(dists_df) == 4  # Flattened array
            assert (
                "distances_px" in dists_df.columns
            )  # Column name should indicate pixels

            mock_load_metrics.assert_called_once_with(
                model_dir.as_posix(), split="test"
            )


class TestUpdateConfigWithWandb:
    """Test suite for update_config_with_wandb function."""

    @patch("sleap_roots_training.train.wandb.config")
    @patch("sleap_roots_training.train.logging")
    def test_update_config_with_wandb(self, mock_logging, mock_wandb_config):
        """Test updating config with W&B parameters."""
        # Mock wandb.config as a dictionary
        mock_config_dict = {
            "data.preprocessing.input_scaling": 0.5,
            "model.backbone.type": "resnet50",
            "training.batch_size": 32,
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

        original_config = {
            "data": {"preprocessing": {"input_scaling": 1.0}},
            "model": {"backbone": {"type": "resnet18"}},
            "training": {"batch_size": 16},
        }

        updated_config = update_config_with_wandb(original_config)

        assert updated_config["data"]["preprocessing"]["input_scaling"] == 0.5
        assert updated_config["model"]["backbone"]["type"] == "resnet50"
        assert updated_config["training"]["batch_size"] == 32

    @patch("sleap_roots_training.train.wandb.config")
    def test_update_config_no_wandb_config(self, mock_wandb_config):
        """Test updating config when W&B config is not available."""
        mock_wandb_config.__bool__ = lambda self: False

        original_config = {"test": "config"}
        updated_config = update_config_with_wandb(original_config)

        assert updated_config == original_config


class TestGetLatestRun:
    """Test suite for get_latest_run function."""

    def test_get_latest_run(self):
        """Test getting latest run directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            models_dir = Path(temp_dir)

            # Create timestamped directories
            (models_dir / "run_241201_120000").mkdir()
            (models_dir / "run_241201_140000").mkdir()
            (models_dir / "run_241201_100000").mkdir()

            latest_run = get_latest_run(models_dir)

            assert latest_run.name == "run_241201_140000"

    def test_get_latest_run_no_valid_dirs(self):
        """Test getting latest run when no valid directories exist."""
        with tempfile.TemporaryDirectory() as temp_dir:
            models_dir = Path(temp_dir)

            # Create non-matching directories
            (models_dir / "invalid_dir").mkdir()
            (models_dir / "another_invalid").mkdir()

            latest_run = get_latest_run(models_dir)

            assert latest_run is None

    def test_get_latest_run_nonexistent_dir(self):
        """Test getting latest run from nonexistent directory."""
        with pytest.raises(FileNotFoundError, match="Models directory not found"):
            get_latest_run(Path("/nonexistent/directory"))


class TestGetParamCombinations:
    """Test suite for get_param_combinations function."""

    def test_get_param_combinations_grid(self):
        """Test parameter combinations calculation for grid search."""
        sweep_config = {
            "method": "grid",
            "parameters": {
                "param1": {"values": [1, 2, 3]},
                "param2": {"values": [0.1, 0.2]},
                "param3": {"values": ["a", "b", "c", "d"]},
            },
        }

        combinations = get_param_combinations(sweep_config)

        assert combinations == 3 * 2 * 4  # 24

    def test_get_param_combinations_random(self):
        """Test parameter combinations for random search."""
        sweep_config = {
            "method": "random",
            "parameters": {"param1": {"values": [1, 2, 3]}},
        }

        combinations = get_param_combinations(sweep_config)

        assert combinations is None

    def test_get_param_combinations_empty_params(self):
        """Test parameter combinations with empty parameters."""
        sweep_config = {"method": "grid", "parameters": {}}

        combinations = get_param_combinations(sweep_config)

        assert combinations == 1  # Empty product is 1


class TestMakeSweepTrainFn:
    """Test suite for make_sweep_train_fn function."""

    @patch("sleap_roots_training.train.wandb.config")
    @patch("sleap_roots_training.train.wandb.init")
    @patch("sleap_roots_training.train.update_config_with_wandb")
    @patch("sleap_roots_training.train.execute_training")
    @patch("sleap_roots_training.train.get_latest_run")
    @patch("sleap_roots_training.train.log_model_artifact_with_evals")
    @patch("builtins.open", new_callable=mock_open, read_data='{"test": "config"}')
    def test_make_sweep_train_fn_success(
        self,
        mock_file,
        mock_log_artifact,
        mock_get_latest_run,
        mock_execute_training,
        mock_update_config,
        mock_wandb_init,
        mock_wandb_config,
    ):
        """Test successful sweep training function creation and execution."""
        # Setup mocks
        mock_run = MagicMock()
        mock_run.id = "test_run_id"
        mock_wandb_init.return_value = mock_run

        mock_updated_config = {"updated": "config"}
        mock_update_config.return_value = mock_updated_config

        mock_model_dir = MagicMock()
        mock_model_dir.exists.return_value = True
        mock_get_latest_run.return_value = mock_model_dir

        # Create sweep training function
        train_fn = make_sweep_train_fn(
            version="1",
            config_copy={"original": "config"},
            dir_path=Path("/fake/dir"),
            sleap_train_command="sleap-train {}",
            experiment_name="test_experiment",
            model_tags=["test_tag"],
            link_to_registry=False,
            registry_name=None,
        )

        # Execute the training function
        train_fn()

        # Verify wandb.init was called with group parameter
        mock_wandb_init.assert_called_once_with(group="test_experiment")

        # Verify config was updated
        mock_update_config.assert_called_once_with({"original": "config"})

        # Verify training was executed
        mock_execute_training.assert_called_once()
        args, kwargs = mock_execute_training.call_args
        assert "sleap-train" in args[0]

        # Verify model artifact was logged
        mock_log_artifact.assert_called_once()

    @patch("sleap_roots_training.train.wandb.config")
    @patch("sleap_roots_training.train.wandb.init")
    @patch("sleap_roots_training.train.update_config_with_wandb")
    @patch("sleap_roots_training.train.execute_training")
    @patch("builtins.open", new_callable=mock_open, read_data='{"test": "config"}')
    def test_make_sweep_train_fn_training_failure(
        self,
        mock_file,
        mock_execute_training,
        mock_update_config,
        mock_wandb_init,
        mock_wandb_config,
    ):
        """Test sweep training function handles training failure."""
        # Setup mocks
        mock_run = MagicMock()
        mock_run.id = "test_run_id"
        mock_wandb_init.return_value = mock_run

        mock_update_config.return_value = {"updated": "config"}
        mock_execute_training.side_effect = Exception("Training failed")

        # Create sweep training function
        train_fn = make_sweep_train_fn(
            version="1",
            config_copy={"original": "config"},
            dir_path=Path("/fake/dir"),
            sleap_train_command="sleap-train {}",
            experiment_name="test_experiment",
            model_tags=["test_tag"],
            link_to_registry=False,
            registry_name=None,
        )

        # Execute the training function and expect exception
        with pytest.raises(Exception, match="Training failed"):
            train_fn()

        # Verify wandb.init was still called with group parameter
        mock_wandb_init.assert_called_once_with(group="test_experiment")

    @patch("sleap_roots_training.train.wandb.config")
    @patch("sleap_roots_training.train.wandb.init")
    @patch("sleap_roots_training.train.update_config_with_wandb")
    @patch("sleap_roots_training.train.execute_training")
    @patch("sleap_roots_training.train.get_latest_run")
    @patch("sleap_roots_training.train.log_model_artifact_with_evals")
    @patch("builtins.open", new_callable=mock_open, read_data='{"test": "config"}')
    def test_make_sweep_train_fn_with_registry(
        self,
        mock_file,
        mock_log_artifact,
        mock_get_latest_run,
        mock_execute_training,
        mock_update_config,
        mock_wandb_init,
        mock_wandb_config,
    ):
        """Test sweep training function with registry linking."""
        # Setup mocks
        mock_run = MagicMock()
        mock_run.id = "test_run_id"
        mock_wandb_init.return_value = mock_run

        mock_update_config.return_value = {"updated": "config"}
        mock_model_dir = MagicMock()
        mock_model_dir.exists.return_value = True
        mock_get_latest_run.return_value = mock_model_dir

        # Create sweep training function with registry
        train_fn = make_sweep_train_fn(
            version="1",
            config_copy={"original": "config"},
            dir_path=Path("/fake/dir"),
            sleap_train_command="sleap-train {}",
            experiment_name="test_experiment",
            model_tags=["test_tag"],
            link_to_registry=True,
            registry_name="test_registry",
        )

        # Execute the training function
        train_fn()

        # Verify log_model_artifact_with_evals was called with registry parameters
        mock_log_artifact.assert_called_once()
        args, kwargs = mock_log_artifact.call_args
        # Check that registry parameters are passed - using positional arguments
        # log_model_artifact_with_evals(run, experiment_name, model_tags, model_dir, version, eval_fn, eval_args, link_to_registry, registry_name)
        assert args[7] == True  # link_to_registry
        assert args[8] == "test_registry"  # registry_name


class TestRunSweepTraining:
    """Test suite for run_sweep_training function."""

    @patch("sleap_roots_training.train.wandb.sweep")
    @patch("sleap_roots_training.train.wandb.agent")
    @patch("sleap_roots_training.train.make_sweep_train_fn")
    @patch("sleap_roots_training.train.get_param_combinations")
    def test_run_sweep_training_basic(
        self,
        mock_get_param_combinations,
        mock_make_sweep_train_fn,
        mock_wandb_agent,
        mock_wandb_sweep,
    ):
        """Test basic sweep training execution."""
        # Setup mocks
        mock_wandb_sweep.return_value = "test_sweep_id"
        mock_get_param_combinations.return_value = 4
        mock_train_fn = MagicMock()
        mock_make_sweep_train_fn.return_value = mock_train_fn

        sweep_config = {
            "method": "grid",
            "parameters": {"param1": {"values": [1, 2]}, "param2": {"values": [3, 4]}},
        }

        # Execute sweep training
        run_sweep_training(
            project_name="test_project",
            entity_name="test_entity",
            experiment_name="test_experiment",
            version="1",
            config_copy={"original": "config"},
            dir_path=Path("/fake/dir"),
            model_tags=["test_tag"],
            sleap_train_command="sleap-train {}",
            sweep_config=sweep_config,
            link_to_registry=False,
            registry_name=None,
        )

        # Verify sweep was created
        mock_wandb_sweep.assert_called_once()
        args, kwargs = mock_wandb_sweep.call_args
        assert kwargs["project"] == "test_project"
        assert kwargs["entity"] == "test_entity"

        # Verify sweep training function was created
        mock_make_sweep_train_fn.assert_called_once()

        # Verify agent was started
        mock_wandb_agent.assert_called_once()
        args, kwargs = mock_wandb_agent.call_args
        assert args[0] == "test_sweep_id"
        assert kwargs["function"] == mock_train_fn
        assert kwargs["count"] == 4

    @patch("sleap_roots_training.train.wandb.sweep")
    @patch("sleap_roots_training.train.wandb.agent")
    @patch("sleap_roots_training.train.make_sweep_train_fn")
    @patch("sleap_roots_training.train.get_param_combinations")
    def test_run_sweep_training_with_registry(
        self,
        mock_get_param_combinations,
        mock_make_sweep_train_fn,
        mock_wandb_agent,
        mock_wandb_sweep,
    ):
        """Test sweep training with registry linking."""
        # Setup mocks
        mock_wandb_sweep.return_value = "test_sweep_id"
        mock_get_param_combinations.return_value = 2
        mock_train_fn = MagicMock()
        mock_make_sweep_train_fn.return_value = mock_train_fn

        sweep_config = {
            "method": "random",
            "parameters": {"param1": {"values": [1, 2]}},
        }

        # Execute sweep training with registry
        run_sweep_training(
            project_name="test_project",
            entity_name="test_entity",
            experiment_name="test_experiment",
            version="1",
            config_copy={"original": "config"},
            dir_path=Path("/fake/dir"),
            model_tags=["test_tag"],
            sleap_train_command="sleap-train {}",
            sweep_config=sweep_config,
            link_to_registry=True,
            registry_name="test_registry",
        )

        # Verify sweep training function was created with registry parameters
        mock_make_sweep_train_fn.assert_called_once()
        args, kwargs = mock_make_sweep_train_fn.call_args
        assert kwargs["link_to_registry"] == True
        assert kwargs["registry_name"] == "test_registry"


class TestGetLatestRunAdditional:
    """Additional tests for get_latest_run function."""

    def test_get_latest_run_multiple_dirs(self):
        """Test get_latest_run with multiple timestamped directories."""
        with tempfile.TemporaryDirectory() as temp_dir:
            models_dir = Path(temp_dir)

            # Create multiple run directories with different timestamps
            dirs = ["run_240101_120000", "run_240102_140000", "run_240102_100000"]
            for d in dirs:
                (models_dir / d).mkdir()

            latest = get_latest_run(models_dir)
            assert latest is not None
            assert latest.name == "run_240102_140000"  # Latest timestamp


class TestRunSweepTrainingAdditional:
    """Additional tests for run_sweep_training function."""

    @patch("sleap_roots_training.train.wandb.sweep")
    @patch("sleap_roots_training.train.wandb.agent")
    @patch("sleap_roots_training.train.make_sweep_train_fn")
    @patch("sleap_roots_training.train.get_param_combinations")
    def test_run_sweep_training_random_method(
        self, mock_get_params, mock_make_fn, mock_agent, mock_sweep
    ):
        """Test sweep with random method (undetermined count)."""
        mock_sweep.return_value = "sweep_id"
        mock_get_params.return_value = None  # Random method returns None
        mock_train_fn = MagicMock()
        mock_make_fn.return_value = mock_train_fn

        sweep_config = {"method": "random", "parameters": {}}

        run_sweep_training(
            project_name="test",
            entity_name="test",
            experiment_name="test",
            version="1",
            config_copy={},
            dir_path=Path("."),
            model_tags=[],
            sleap_train_command="",
            sweep_config=sweep_config,
            link_to_registry=False,
            registry_name=None,
        )

        # Verify agent was called without count
        call_kwargs = mock_agent.call_args[1]
        assert "count" not in call_kwargs or call_kwargs["count"] is None


class TestMakeSweepTrainFnAdditional:
    """Additional tests for make_sweep_train_fn."""

    @patch("sleap_roots_training.train.wandb")
    @patch("sleap_roots_training.train.update_config_with_wandb")
    @patch("sleap_roots_training.train.execute_training")
    @patch("sleap_roots_training.train.get_latest_run")
    @patch("sleap_roots_training.train.logging")
    @patch("builtins.open", new_callable=mock_open, read_data='{"test": "config"}')
    def test_sweep_train_fn_no_model_dir(
        self,
        mock_file,
        mock_logging,
        mock_get_latest_run,
        mock_execute_training,
        mock_update_config,
        mock_wandb,
    ):
        """Test sweep train function when model directory is not found."""
        # Setup mocks
        mock_wandb.run = MagicMock()
        mock_wandb.run.id = "test_run_id"
        mock_wandb.init.return_value = MagicMock()
        mock_wandb.config = {}

        mock_update_config.return_value = {"updated": "config"}
        mock_get_latest_run.return_value = None  # No model dir found

        # Create sweep training function
        train_fn = make_sweep_train_fn(
            version="1",
            config_copy={"original": "config"},
            dir_path=Path("/fake/dir"),
            sleap_train_command="sleap-train {}",
            experiment_name="test_experiment",
            model_tags=["test_tag"],
            link_to_registry=False,
            registry_name=None,
        )

        # Execute and expect error
        with pytest.raises(FileNotFoundError):
            train_fn()

        # Verify error was logged
        mock_logging.error.assert_called()


class TestMainFunction:
    """Test suite for main function."""

    @patch("sleap_roots_training.train.CONFIG")
    @patch("sleap_roots_training.train.load_training_data")
    @patch("sleap_roots_training.train.run_single_training")
    def test_main_single_training(self, mock_run_single, mock_load_data, mock_config):
        """Test main function with single training."""
        # Setup mocks
        mock_config.__getitem__.side_effect = lambda key: {
            "project_name": "test_project",
            "entity_name": "test_entity",
            "experiment_name": "test_experiment",
            "registry": "test_registry",
        }[key]
        mock_config.get.return_value = "test_registry"

        mock_df = pd.DataFrame(
            {
                "version": [1, 2],
                "path": ["/path/to/v1/config.json", "/path/to/v2/config.json"],
            }
        )
        mock_load_data.return_value = mock_df

        # Create temporary config files
        with tempfile.TemporaryDirectory() as temp_dir:
            for version in [1, 2]:
                config_dir = Path(temp_dir) / f"v{version}"
                config_dir.mkdir()
                config_path = config_dir / f"initial_config_modified_v00{version}.json"
                with open(config_path, "w") as f:
                    json.dump({"test": f"config{version}"}, f)

                # Update mock dataframe paths
                mock_df.loc[mock_df["version"] == version, "path"] = str(config_path)

            main(
                csv_path="test.csv",
                tags=["test"],
                model_tags=["model"],
                use_existing_model=False,
                use_sweep=False,
                link_to_registry=True,
            )

            # Should call run_single_training for each version
            assert mock_run_single.call_count == 2

    @patch("sleap_roots_training.train.CONFIG")
    @patch("sleap_roots_training.train.load_training_data")
    @patch("sleap_roots_training.train.run_sweep_training")
    def test_main_sweep_training(self, mock_run_sweep, mock_load_data, mock_config):
        """Test main function with sweep training."""
        # Setup mocks
        mock_config.__getitem__.side_effect = lambda key: {
            "project_name": "test_project",
            "entity_name": "test_entity",
            "experiment_name": "test_experiment",
            "registry": "test_registry",
        }[key]
        mock_config.get.return_value = "test_registry"

        mock_df = pd.DataFrame({"version": [1], "path": ["/path/to/v1/config.json"]})
        mock_load_data.return_value = mock_df

        sweep_config = {"method": "grid", "parameters": {"param1": {"values": [1, 2]}}}

        # Create temporary config file
        with tempfile.TemporaryDirectory() as temp_dir:
            config_dir = Path(temp_dir) / "v1"
            config_dir.mkdir()
            config_path = config_dir / "initial_config_modified_v001.json"
            with open(config_path, "w") as f:
                json.dump({"test": "config1"}, f)

            mock_df.loc[0, "path"] = str(config_path)

            main(
                csv_path="test.csv",
                use_sweep=True,
                sweep_config=sweep_config,
                link_to_registry=True,
            )

            # Should call run_sweep_training for each version
            assert mock_run_sweep.call_count == 1

    @patch("sleap_roots_training.train.CONFIG")
    @patch("sleap_roots_training.train.load_training_data")
    def test_main_missing_config_file(self, mock_load_data, mock_config):
        """Test main function with missing config file."""
        # Setup mocks
        mock_config.__getitem__.side_effect = lambda key: {
            "project_name": "test_project",
            "entity_name": "test_entity",
            "experiment_name": "test_experiment",
            "registry": "test_registry",
        }[key]
        mock_config.get.return_value = "test_registry"

        mock_df = pd.DataFrame(
            {"version": [1], "path": ["/nonexistent/path/config.json"]}
        )
        mock_load_data.return_value = mock_df

        with pytest.raises(FileNotFoundError, match="Config file not found"):
            main(csv_path="test.csv", use_sweep=False, link_to_registry=True)

    @patch("builtins.open", new_callable=mock_open, read_data='{"test": "config"}')
    @patch("sleap_roots_training.train.Path.exists")
    @patch("sleap_roots_training.train.CONFIG")
    @patch("sleap_roots_training.train.load_training_data")
    def test_main_sweep_without_config(
        self, mock_load_data, mock_config, mock_exists, mock_file
    ):
        """Test main function with sweep but no sweep config."""
        # Setup mocks
        mock_config.__getitem__.side_effect = lambda key: {
            "project_name": "test_project",
            "entity_name": "test_entity",
            "experiment_name": "test_experiment",
            "registry": "test_registry",
        }[key]

        mock_df = pd.DataFrame({"version": [1], "path": ["/path/to/v1/config.json"]})
        mock_load_data.return_value = mock_df

        # Mock file existence to avoid FileNotFoundError
        mock_exists.return_value = True

        with pytest.raises(ValueError, match="Sweep config must be provided"):
            main(
                csv_path="test.csv",
                use_sweep=True,
                sweep_config=None,
                link_to_registry=True,
            )


class TestRealDataCoverage:
    """Tests using real data files to improve coverage."""

    def test_real_model_directory_structure(self):
        """Test functions with real model directory structure."""
        test_data_dir = (
            Path(__file__).parent / "data" / "min_tracks_2node.UNet.bottomup_multiclass"
        )

        # Test that the directory exists and has expected files
        assert test_data_dir.exists()
        assert (test_data_dir / "initial_config.json").exists()
        assert (test_data_dir / "training_config.json").exists()
        assert (test_data_dir / "best_model.h5").exists()

    def test_get_param_combinations_realistic(self):
        """Test parameter combinations with realistic sweep configs."""
        # Test realistic grid search config
        realistic_config = {
            "method": "grid",
            "parameters": {
                "data.preprocessing.input_scaling": {"values": [0.5, 1.0, 1.5]},
                "model.backbone.unet.filters": {"values": [8, 16, 32]},
                "model.backbone.unet.max_stride": {"values": [16, 32]},
                "optimization.batch_size": {"values": [2, 4]},
            },
        }

        combinations = get_param_combinations(realistic_config)
        expected = 3 * 3 * 2 * 2  # 36 combinations
        assert combinations == expected

    def test_get_param_combinations_edge_cases(self):
        """Test parameter combinations edge cases."""
        # Empty parameters
        empty_config = {"method": "grid", "parameters": {}}
        assert get_param_combinations(empty_config) == 1

        # Single parameter
        single_config = {
            "method": "grid",
            "parameters": {"param1": {"values": [1, 2, 3]}},
        }
        assert get_param_combinations(single_config) == 3

        # Random method (should return None)
        random_config = {
            "method": "random",
            "parameters": {"param1": {"values": [1, 2, 3]}},
        }
        assert get_param_combinations(random_config) is None

    def test_get_latest_run_realistic_structure(self):
        """Test get_latest_run with realistic directory structures."""
        with tempfile.TemporaryDirectory() as temp_dir:
            models_dir = Path(temp_dir)

            # Create realistic run directory structure
            run_dirs = [
                "run_250115_120000",
                "run_250116_143000",
                "run_250115_180000",
            ]

            for run_dir in run_dirs:
                dir_path = models_dir / run_dir
                dir_path.mkdir()
                # Add some realistic files
                (dir_path / "best_model.h5").touch()
                (dir_path / "metrics.json").touch()

            latest_run = get_latest_run(models_dir)
            assert latest_run is not None
            assert latest_run.name == "run_250116_143000"

    def test_get_latest_run_no_valid_dirs(self):
        """Test get_latest_run with no valid run directories."""
        with tempfile.TemporaryDirectory() as temp_dir:
            models_dir = Path(temp_dir)

            # Create directories that don't match pattern
            (models_dir / "not_a_run").mkdir()
            (models_dir / "also_not_run").mkdir()
            (models_dir / "run_invalid").mkdir()  # Doesn't match timestamp pattern

            result = get_latest_run(models_dir)
            assert result is None

    def test_get_latest_run_nonexistent_directory(self):
        """Test get_latest_run with nonexistent directory."""
        nonexistent = Path("/definitely/does/not/exist")

        with pytest.raises(FileNotFoundError, match="Models directory not found"):
            get_latest_run(nonexistent)

    def test_empty_data_structures(self):
        """Test functions with empty data structures."""
        # Test empty sweep config
        empty_config = {"method": "grid", "parameters": {}}
        result = get_param_combinations(empty_config)
        assert result == 1

    def test_load_training_data_valid_csv(self):
        """Test loading valid CSV training data."""
        with tempfile.TemporaryDirectory() as temp_dir:
            csv_path = Path(temp_dir) / "test_data.csv"
            test_data = pd.DataFrame(
                {
                    "version": [1, 2],
                    "path": ["config1.json", "config2.json"],
                    "labeled_frames": [100, 150],
                    "split_type": ["train", "val"],
                }
            )
            test_data.to_csv(csv_path, index=False)

            result = load_training_data(str(csv_path))

            assert isinstance(result, pd.DataFrame)
            assert len(result) == 2
            assert "version" in result.columns

    def test_get_training_groups_basic(self):
        """Test basic training group extraction."""
        df = pd.DataFrame({"version": [1, 1, 2], "path": ["path1", "path2", "path3"]})

        result = get_training_groups(df)

        # It returns a GroupBy object
        assert hasattr(result, "groups")
        assert len(result.groups) == 2  # Two unique versions
        assert 1 in result.groups
        assert 2 in result.groups


class TestAdditionalTrainFunctions:
    """Additional tests for train.py functions."""

    @patch("sleap_roots_training.train.subprocess.run")
    def test_execute_training_success(self, mock_run):
        """Test successful training execution."""
        mock_run.return_value = MagicMock(returncode=0)

        command = "sleap-train config.json"
        execute_training(command)

        mock_run.assert_called_once()

    @patch("sleap_roots_training.train.subprocess.run")
    def test_execute_training_failure(self, mock_run):
        """Test training execution with failure."""
        from subprocess import CalledProcessError

        mock_run.side_effect = CalledProcessError(1, "command", stderr="Error message")

        command = "sleap-train invalid_config.json"

        # The function should re-raise the exception after logging
        with pytest.raises(CalledProcessError):
            execute_training(command)

        mock_run.assert_called_once()

    @patch("sleap_roots_training.train.wandb")
    def test_update_config_basic(self, mock_wandb):
        """Test basic config update with wandb parameters."""
        mock_wandb.config = {
            "data.preprocessing.input_scaling": 1.5,
            "model.backbone.unet.filters": 16,
            "optimization.batch_size": 8,
        }

        base_config = {
            "data": {"preprocessing": {"input_scaling": 0.5}},
            "model": {"backbone": {"unet": {"filters": 8}}},
            "optimization": {"batch_size": 4},
        }

        result = update_config_with_wandb(base_config)

        assert result["data"]["preprocessing"]["input_scaling"] == 1.5
        assert result["model"]["backbone"]["unet"]["filters"] == 16
        assert result["optimization"]["batch_size"] == 8

    @patch("sleap_roots_training.train.wandb")
    def test_update_config_nested_creation(self, mock_wandb):
        """Test config update that creates nested structures."""
        mock_wandb.config = {
            "new.nested.parameter": "test_value",
            "model.head.new_param": 42,
        }

        base_config = {"existing": "value"}

        result = update_config_with_wandb(base_config)

        assert result["new"]["nested"]["parameter"] == "test_value"
        assert result["model"]["head"]["new_param"] == 42
        assert result["existing"] == "value"


class TestRunSingleTrainingCoverage:
    """Tests to improve coverage of run_single_training function."""

    @patch("sleap_roots_training.train.wandb.config", {})
    @patch("sleap_roots_training.train.CONFIG")
    @patch("sleap_roots_training.train.log_to_wandb")
    @patch("sleap_roots_training.train.execute_training")
    @patch("sleap_roots_training.train.get_latest_run")
    @patch("sleap_roots_training.train.log_model_artifact_with_evals")
    def test_run_single_training_basic_coverage(
        self,
        mock_log_artifact,
        mock_get_latest,
        mock_execute,
        mock_log_wandb,
        mock_config,
    ):
        """Test run_single_training to improve coverage."""
        # Setup mocks
        mock_config.__getitem__.side_effect = lambda key: {
            "project_name": "test_project",
            "entity_name": "test_entity",
        }[key]

        mock_run = MagicMock()
        mock_run.id = "test_run_123"
        mock_log_wandb.return_value = mock_run

        mock_model_dir = MagicMock()
        mock_model_dir.exists.return_value = True
        mock_get_latest.return_value = mock_model_dir

        # Create temporary config file
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "config.json"
            config_data = {"test": "config"}
            with open(config_path, "w") as f:
                json.dump(config_data, f)

            # Call the function
            run_single_training(
                project_name="test_project",
                entity_name="test_entity",
                experiment_name="test_exp",
                version="1",
                config_path=config_path,
                config_copy=config_data,
                dir_path=Path(temp_dir),
                model_tags=["test"],
                tags=["integration"],
                sleap_train_command="sleap-train {}",
                use_existing_model=False,
                link_to_registry=True,
                registry_name="test_registry",
            )

        # Verify function calls
        mock_log_wandb.assert_called_once()
        mock_execute.assert_called_once()
        mock_get_latest.assert_called_once()
        mock_log_artifact.assert_called_once()
        mock_run.finish.assert_called_once()

    @patch("sleap_roots_training.train.wandb.config", {})
    @patch("sleap_roots_training.train.CONFIG")
    @patch("sleap_roots_training.train.log_to_wandb")
    @patch("sleap_roots_training.train.execute_training")
    @patch("sleap_roots_training.train.get_latest_run")
    def test_run_single_training_no_model_found(
        self, mock_get_latest, mock_execute, mock_log_wandb, mock_config
    ):
        """Test run_single_training when no model directory is found."""
        # Setup mocks
        mock_config.__getitem__.side_effect = lambda key: {
            "project_name": "test_project",
            "entity_name": "test_entity",
        }[key]

        mock_run = MagicMock()
        mock_log_wandb.return_value = mock_run
        mock_get_latest.return_value = None  # No model found

        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "config.json"
            with open(config_path, "w") as f:
                json.dump({"test": "config"}, f)

            # Should raise FileNotFoundError when no model found
            with pytest.raises(
                FileNotFoundError, match="No existing model directory found"
            ):
                run_single_training(
                    project_name="test_project",
                    entity_name="test_entity",
                    experiment_name="test_exp",
                    version="1",
                    config_path=config_path,
                    config_copy={"test": "config"},
                    dir_path=Path(temp_dir),
                    model_tags=["test"],
                    tags=["test"],
                    sleap_train_command="sleap-train {}",
                    use_existing_model=False,
                    link_to_registry=False,
                    registry_name=None,
                )

        mock_run.finish.assert_called_once()


class TestLogModelArtifactWithEvalsCoverage:
    """Tests to improve coverage of log_model_artifact_with_evals function."""

    @patch("sleap_roots_training.train.wandb.Artifact")
    def test_log_model_artifact_with_evals_basic(self, mock_artifact_class):
        """Test basic functionality of log_model_artifact_with_evals."""
        mock_run = MagicMock()
        mock_artifact = MagicMock()
        mock_artifact_class.return_value = mock_artifact

        # Setup mock so run.log_artifact returns a mock logged artifact with link method
        mock_logged_artifact = MagicMock()
        mock_run.log_artifact.return_value = mock_logged_artifact

        # Mock eval function
        mock_eval_fn = MagicMock()
        mock_metrics_df = MagicMock()
        mock_dists_df = MagicMock()
        mock_visualizations = {"hist": "path.png"}
        mock_eval_fn.return_value = (
            mock_metrics_df,
            mock_dists_df,
            mock_visualizations,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            model_dir = Path(temp_dir) / "model"
            model_dir.mkdir()

            # Create a training config file
            config_path = model_dir / "training_config.json"
            config_data = {"model": {"type": "test"}, "data": {"test": "config"}}
            with open(config_path, "w") as f:
                json.dump(config_data, f)

            log_model_artifact_with_evals(
                run=mock_run,
                experiment_name="test_exp",
                model_tags=["test"],
                model_dir=model_dir,
                version="1",
                eval_fn=mock_eval_fn,
                eval_args={"px_per_mm": 15.0},
                link_to_registry=True,
                registry_name="test_registry",
            )

        # Verify calls
        mock_eval_fn.assert_called_once()
        mock_artifact_class.assert_called()
        mock_run.log_artifact.assert_called()
        mock_logged_artifact.link.assert_called_once_with(
            "model-registry/test_registry"
        )
        mock_run.config.update.assert_called_once_with(config_data)

    def test_log_model_artifact_missing_registry_name(self):
        """Test error when registry linking requested but no registry name provided."""
        mock_run = MagicMock()

        with tempfile.TemporaryDirectory() as temp_dir:
            model_dir = Path(temp_dir) / "model"
            model_dir.mkdir()

            with pytest.raises(
                ValueError,
                match="registry_name must be provided when link_to_registry is True",
            ):
                log_model_artifact_with_evals(
                    run=mock_run,
                    experiment_name="test_exp",
                    model_tags=["test"],
                    model_dir=model_dir,
                    version="1",
                    eval_fn=None,
                    eval_args={},
                    link_to_registry=True,
                    registry_name=None,  # Missing registry name
                )
