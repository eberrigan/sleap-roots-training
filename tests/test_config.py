import pytest
import tempfile
import yaml
from pathlib import Path
from unittest.mock import patch

from sleap_roots_training.config import (
    create_default_config,
    load_config,
    save_config,
    update_config,
    reset_config,
    DEFAULT_CONFIG,
    CONFIG_PATH,
)


class TestConfig:
    """Test suite for config module."""

    def test_default_config_structure(self):
        """Test that DEFAULT_CONFIG has expected structure."""
        expected_keys = {
            "project_name",
            "entity_name",
            "experiment_name",
            "registry",
            "collection_name",
            "job_type",
        }
        assert set(DEFAULT_CONFIG.keys()) == expected_keys
        assert DEFAULT_CONFIG["project_name"] == "sleap-roots"
        assert (
            DEFAULT_CONFIG["entity_name"]
            == "eberrigan-salk-institute-for-biological-studies"
        )
        assert DEFAULT_CONFIG["experiment_name"] is None
        assert DEFAULT_CONFIG["registry"] is None
        assert DEFAULT_CONFIG["collection_name"] is None
        assert DEFAULT_CONFIG["job_type"] is None

    def test_create_default_config(self):
        """Test creating default config file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_config_path = Path(temp_dir) / "test_config.yaml"

            with patch("sleap_roots_training.config.CONFIG_PATH", test_config_path):
                create_default_config()

                assert test_config_path.exists()
                with open(test_config_path, "r") as f:
                    loaded_config = yaml.safe_load(f)
                assert loaded_config == DEFAULT_CONFIG

    def test_create_default_config_existing_file(self):
        """Test that create_default_config doesn't overwrite existing file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_config_path = Path(temp_dir) / "test_config.yaml"

            # Create existing config file
            existing_config = {"test": "value"}
            with open(test_config_path, "w") as f:
                yaml.safe_dump(existing_config, f)

            with patch("sleap_roots_training.config.CONFIG_PATH", test_config_path):
                create_default_config()

                # File should still contain original content
                with open(test_config_path, "r") as f:
                    loaded_config = yaml.safe_load(f)
                assert loaded_config == existing_config

    def test_load_config_existing_file(self):
        """Test loading config from existing file."""
        test_config = {"project_name": "test_project", "entity_name": "test_entity"}

        with tempfile.TemporaryDirectory() as temp_dir:
            test_config_path = Path(temp_dir) / "test_config.yaml"
            with open(test_config_path, "w") as f:
                yaml.safe_dump(test_config, f)

            with patch("sleap_roots_training.config.CONFIG_PATH", test_config_path):
                loaded_config = load_config()
                assert loaded_config == test_config

    def test_load_config_nonexistent_file(self):
        """Test loading config creates default when file doesn't exist."""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_config_path = Path(temp_dir) / "nonexistent_config.yaml"

            with patch("sleap_roots_training.config.CONFIG_PATH", test_config_path):
                loaded_config = load_config()
                assert loaded_config == DEFAULT_CONFIG
                assert test_config_path.exists()

    def test_save_config(self):
        """Test saving config to file."""
        test_config = {"project_name": "test_project", "entity_name": "test_entity"}

        with tempfile.TemporaryDirectory() as temp_dir:
            test_config_path = Path(temp_dir) / "test_config.yaml"

            with patch("sleap_roots_training.config.CONFIG_PATH", test_config_path):
                save_config(test_config)

                assert test_config_path.exists()
                with open(test_config_path, "r") as f:
                    loaded_config = yaml.safe_load(f)
                assert loaded_config == test_config

    def test_update_config(self):
        """Test updating specific config values."""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_config_path = Path(temp_dir) / "test_config.yaml"

            with patch("sleap_roots_training.config.CONFIG_PATH", test_config_path):
                with patch(
                    "sleap_roots_training.config.CONFIG", DEFAULT_CONFIG.copy()
                ) as mock_config:
                    update_config(project_name="new_project", entity_name="new_entity")

                    assert mock_config["project_name"] == "new_project"
                    assert mock_config["entity_name"] == "new_entity"
                    assert test_config_path.exists()

    def test_reset_config(self):
        """Test resetting config to defaults."""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_config_path = Path(temp_dir) / "test_config.yaml"

            with patch("sleap_roots_training.config.CONFIG_PATH", test_config_path):
                # Create a modified config file first
                modified_config = {"modified": "value"}
                with open(test_config_path, "w") as f:
                    yaml.dump(modified_config, f)

                with patch("sleap_roots_training.config.CONFIG", modified_config):
                    reset_config()

                    # Check that the config file was updated with default values
                    assert test_config_path.exists()
                    with open(test_config_path, "r") as f:
                        saved_config = yaml.safe_load(f)
                    assert saved_config == DEFAULT_CONFIG

    def test_config_path_points_to_correct_location(self):
        """Test that CONFIG_PATH points to the expected location."""
        expected_path = (
            Path(__file__).parent.parent / "sleap_roots_training" / "config.yaml"
        )
        # Normalize paths for comparison
        assert CONFIG_PATH.resolve() == expected_path.resolve()

    @patch("builtins.print")
    def test_create_default_config_prints_message(self, mock_print):
        """Test that create_default_config prints confirmation message."""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_config_path = Path(temp_dir) / "test_config.yaml"

            with patch("sleap_roots_training.config.CONFIG_PATH", test_config_path):
                create_default_config()

                mock_print.assert_called_once_with(
                    f"Default config.yaml created at {test_config_path}"
                )

    @patch("builtins.print")
    def test_save_config_prints_message(self, mock_print):
        """Test that save_config prints confirmation message."""
        test_config = {"test": "value"}

        with tempfile.TemporaryDirectory() as temp_dir:
            test_config_path = Path(temp_dir) / "test_config.yaml"

            with patch("sleap_roots_training.config.CONFIG_PATH", test_config_path):
                save_config(test_config)

                mock_print.assert_called_once_with(
                    f"Configuration updated successfully at {test_config_path}."
                )

    @patch("builtins.print")
    def test_update_config_prints_message(self, mock_print):
        """Test that update_config prints confirmation message."""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_config_path = Path(temp_dir) / "test_config.yaml"

            with patch("sleap_roots_training.config.CONFIG_PATH", test_config_path):
                with patch(
                    "sleap_roots_training.config.CONFIG", DEFAULT_CONFIG.copy()
                ) as mock_config:
                    update_config(project_name="new_project")

                    mock_print.assert_called_with(f"CONFIG updated to {mock_config}.")

    @patch("builtins.print")
    def test_reset_config_prints_message(self, mock_print):
        """Test that reset_config prints confirmation message."""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_config_path = Path(temp_dir) / "test_config.yaml"

            with patch("sleap_roots_training.config.CONFIG_PATH", test_config_path):
                with patch("sleap_roots_training.config.CONFIG", {"modified": "value"}):
                    reset_config()

                    mock_print.assert_called_with(
                        "Configuration has been reset to default values."
                    )
