"""
Comprehensive tests for the new OmegaConf/Hydra-based configuration system.

This test suite covers:
- Configuration loading and validation
- Dynamic configuration updates
- Thread safety and concurrent access
- Error handling and edge cases  
- Backward compatibility
- Config callbacks and event system
- File I/O operations
- Migration from old config system
- Performance and memory usage
- Integration with Hydra
"""

import concurrent.futures
import os
import pytest
import tempfile
import threading
import time
import yaml
from pathlib import Path
from unittest.mock import MagicMock, patch
from omegaconf import DictConfig, OmegaConf

from sleap_roots_training.config_new import (
    ConfigManager,
    ConfigurationError,
    LegacyConfig,
    WandBConfig,
    PathConfig,
    ValidationConfig,
    SleapRootsConfig,
    get_config,
    update_config,
    get_value,
    set_value,
    register_config_callback,
    unregister_config_callback,
    reset_config,
    save_config,
    load_config_from_file,
    migrate_from_old_config,
    CONFIG,
)


@pytest.fixture
def clean_config_manager():
    """
    Provide a clean ConfigManager instance for each test.

    This fixture ensures test isolation by resetting the singleton
    and clearing any Hydra state between tests.
    """
    # Reset singleton instance
    ConfigManager._instance = None

    # Clear any existing Hydra instance
    from hydra.core.global_hydra import GlobalHydra

    if GlobalHydra().is_initialized():
        GlobalHydra.instance().clear()

    # Reset the global config manager in the module
    import sleap_roots_training.config_new as config_new_module

    new_manager = ConfigManager()
    config_new_module._config_manager = new_manager

    yield new_manager

    # Cleanup after test
    ConfigManager._instance = None
    if GlobalHydra().is_initialized():
        GlobalHydra.instance().clear()

    # Reset the global config manager again
    config_new_module._config_manager = ConfigManager()


@pytest.fixture
def sample_config_dict():
    """Sample configuration dictionary for testing."""
    return {
        "wandb": {
            "project_name": "test-project",
            "entity_name": "test-entity",
            "experiment_name": "test-experiment",
            "registry": "test-registry",
            "collection_name": "test-collection",
            "job_type": "test-job",
        },
        "paths": {
            "config_dir": "/tmp/test_config",
            "data_dir": "/tmp/test_data",
            "models_dir": "/tmp/test_models",
            "logs_dir": "/tmp/test_logs",
        },
        "validation": {
            "strict_mode": True,
            "validate_paths": False,  # Disable for testing
            "validate_wandb": True,
            "allow_missing_optional": True,
        },
        "custom": {"test_key": "test_value"},
        "version": "2.0.0",
    }


@pytest.fixture
def temp_config_file(sample_config_dict):
    """Create a temporary configuration file."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        OmegaConf.save(sample_config_dict, f)
        temp_path = f.name

    yield Path(temp_path)

    # Cleanup
    try:
        Path(temp_path).unlink()
    except FileNotFoundError:
        pass


@pytest.fixture
def old_config_file():
    """Create an old-style config.yaml file for migration testing."""
    old_config = {
        "project_name": "old-project",
        "entity_name": "old-entity",
        "experiment_name": "old-experiment",
        "registry": "old-registry",
        "collection_name": "old-collection",
        "job_type": "old-job",
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.safe_dump(old_config, f)
        temp_path = f.name

    yield Path(temp_path)

    # Cleanup
    try:
        Path(temp_path).unlink()
    except FileNotFoundError:
        pass


@pytest.fixture
def mock_callback():
    """Mock callback function for testing config change notifications."""
    return MagicMock()


class TestSleapRootsConfig:
    """Test the structured configuration dataclasses."""

    def test_wandb_config_defaults(self):
        """Test WandBConfig default values."""
        config = WandBConfig()
        assert config.project_name == "sleap-roots"
        assert config.entity_name == "eberrigan-salk-institute-for-biological-studies"
        assert config.experiment_name is None
        assert config.registry is None
        assert config.collection_name is None
        assert config.job_type is None

    def test_wandb_config_custom_values(self):
        """Test WandBConfig with custom values."""
        config = WandBConfig(
            project_name="custom-project",
            entity_name="custom-entity",
            experiment_name="custom-experiment",
        )
        assert config.project_name == "custom-project"
        assert config.entity_name == "custom-entity"
        assert config.experiment_name == "custom-experiment"

    def test_path_config_defaults(self):
        """Test PathConfig default values."""
        config = PathConfig()
        assert config.config_dir == "${oc.env:HOME}/.sleap_roots_training"
        assert config.data_dir is None
        assert config.models_dir is None
        assert config.logs_dir is None

    def test_validation_config_defaults(self):
        """Test ValidationConfig default values."""
        config = ValidationConfig()
        assert config.strict_mode is True
        assert config.validate_paths is True
        assert config.validate_wandb is True
        assert config.allow_missing_optional is True

    def test_sleap_roots_config_structure(self):
        """Test complete SleapRootsConfig structure."""
        config = SleapRootsConfig()
        assert isinstance(config.wandb, WandBConfig)
        assert isinstance(config.paths, PathConfig)
        assert isinstance(config.validation, ValidationConfig)
        assert isinstance(config.custom, dict)
        assert config.version == "2.0.0"


class TestConfigManager:
    """Test the ConfigManager singleton class."""

    def test_singleton_pattern(self, clean_config_manager):
        """Test that ConfigManager follows singleton pattern."""
        manager1 = ConfigManager()
        manager2 = ConfigManager()
        assert manager1 is manager2

    def test_thread_safety_singleton(self, clean_config_manager):
        """Test singleton pattern is thread-safe."""
        managers = []

        def create_manager():
            managers.append(ConfigManager())

        threads = [threading.Thread(target=create_manager) for _ in range(10)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        # All managers should be the same instance
        assert all(manager is managers[0] for manager in managers)

    def test_load_default_config(self, clean_config_manager):
        """Test loading default configuration."""
        manager = clean_config_manager
        config = manager.load_config()

        assert isinstance(config, DictConfig)
        assert config.wandb.project_name == "sleap-roots"
        assert config.version == "2.0.0"

    def test_get_config_loads_default_if_none(self, clean_config_manager):
        """Test get_config loads default if no config is loaded."""
        manager = clean_config_manager
        config = manager.get_config()

        assert isinstance(config, DictConfig)
        assert config.wandb.project_name == "sleap-roots"

    def test_update_config_single_value(self, clean_config_manager):
        """Test updating a single configuration value."""
        manager = clean_config_manager
        manager.load_config()

        manager.update_config(**{"wandb.project_name": "updated-project"})

        config = manager.get_config()
        assert config.wandb.project_name == "updated-project"

    def test_update_config_multiple_values(self, clean_config_manager):
        """Test updating multiple configuration values."""
        manager = clean_config_manager
        manager.load_config()

        updates = {
            "wandb.project_name": "multi-project",
            "wandb.entity_name": "multi-entity",
            "wandb.experiment_name": "multi-experiment",
        }
        manager.update_config(**updates)

        config = manager.get_config()
        assert config.wandb.project_name == "multi-project"
        assert config.wandb.entity_name == "multi-entity"
        assert config.wandb.experiment_name == "multi-experiment"

    def test_update_config_nested_values(self, clean_config_manager):
        """Test updating nested configuration values."""
        manager = clean_config_manager
        manager.load_config()

        manager.update_config(**{"custom.nested.deep.value": "deep-value"})

        config = manager.get_config()
        assert config.custom.nested.deep.value == "deep-value"

    def test_get_value_existing_key(self, clean_config_manager):
        """Test getting value for existing key."""
        manager = clean_config_manager
        manager.load_config()

        value = manager.get_value("wandb.project_name")
        assert value == "sleap-roots"

    def test_get_value_nonexistent_key_with_default(self, clean_config_manager):
        """Test getting value for nonexistent key with default."""
        manager = clean_config_manager
        manager.load_config()

        value = manager.get_value("nonexistent.key", "default-value")
        assert value == "default-value"

    def test_get_value_nonexistent_key_without_default(self, clean_config_manager):
        """Test getting value for nonexistent key without default."""
        manager = clean_config_manager
        manager.load_config()

        value = manager.get_value("nonexistent.key")
        assert value is None

    def test_set_value(self, clean_config_manager):
        """Test setting configuration value by key path."""
        manager = clean_config_manager
        manager.load_config()

        manager.set_value("wandb.project_name", "set-project")

        assert manager.get_value("wandb.project_name") == "set-project"

    def test_config_callbacks_registration(self, clean_config_manager, mock_callback):
        """Test registering configuration change callbacks."""
        manager = clean_config_manager
        manager.load_config()

        manager.register_callback(mock_callback)
        manager.update_config(**{"wandb.project_name": "callback-project"})

        mock_callback.assert_called_once()
        called_config = mock_callback.call_args[0][0]
        assert called_config.wandb.project_name == "callback-project"

    def test_config_callbacks_unregistration(self, clean_config_manager, mock_callback):
        """Test unregistering configuration change callbacks."""
        manager = clean_config_manager
        manager.load_config()

        manager.register_callback(mock_callback)
        manager.unregister_callback(mock_callback)
        manager.update_config(**{"wandb.project_name": "callback-project"})

        mock_callback.assert_not_called()

    def test_config_callbacks_multiple(self, clean_config_manager):
        """Test multiple configuration change callbacks."""
        manager = clean_config_manager
        manager.load_config()

        callback1 = MagicMock()
        callback2 = MagicMock()

        manager.register_callback(callback1)
        manager.register_callback(callback2)
        manager.update_config(**{"wandb.project_name": "multi-callback"})

        callback1.assert_called_once()
        callback2.assert_called_once()

    def test_config_callbacks_exception_handling(self, clean_config_manager):
        """Test that callback exceptions don't break config updates."""
        manager = clean_config_manager
        manager.load_config()

        def failing_callback(config):
            raise Exception("Callback failed")

        manager.register_callback(failing_callback)

        # Should not raise exception
        manager.update_config(**{"wandb.project_name": "exception-test"})

        # Config should still be updated
        assert manager.get_value("wandb.project_name") == "exception-test"

    def test_reset_to_defaults(self, clean_config_manager):
        """Test resetting configuration to defaults."""
        manager = clean_config_manager
        manager.load_config()

        # Modify config
        manager.update_config(**{"wandb.project_name": "modified-project"})
        assert manager.get_value("wandb.project_name") == "modified-project"

        # Reset to defaults
        manager.reset_to_defaults()
        assert manager.get_value("wandb.project_name") == "sleap-roots"

    def test_to_dict(self, clean_config_manager):
        """Test converting configuration to dictionary."""
        manager = clean_config_manager
        manager.load_config()

        config_dict = manager.to_dict()

        assert isinstance(config_dict, dict)
        assert config_dict["wandb"]["project_name"] == "sleap-roots"
        assert config_dict["version"] == "2.0.0"

    def test_save_and_load_config(self, clean_config_manager, temp_config_file):
        """Test saving and loading configuration from file."""
        manager = clean_config_manager
        manager.load_config()

        # Modify config
        manager.update_config(**{"wandb.project_name": "save-test"})

        # Save to file
        save_path = temp_config_file.with_suffix(".save.yaml")
        manager.save_config(save_path)
        assert save_path.exists()

        # Reset and load from file
        manager.reset_to_defaults()
        assert manager.get_value("wandb.project_name") == "sleap-roots"

        loaded_config = manager.load_from_file(save_path)
        assert loaded_config.wandb.project_name == "save-test"

        # Cleanup
        try:
            save_path.unlink()
        except FileNotFoundError:
            pass


class TestConfigValidation:
    """Test configuration validation functionality."""

    def test_validation_wandb_missing_project_name(self, clean_config_manager):
        """Test validation fails when project_name is missing."""
        manager = clean_config_manager

        with pytest.raises(ConfigurationError, match="project_name is required"):
            manager.update_config(**{"wandb.project_name": ""})

    def test_validation_wandb_missing_entity_name(self, clean_config_manager):
        """Test validation fails when entity_name is missing."""
        manager = clean_config_manager

        with pytest.raises(ConfigurationError, match="entity_name is required"):
            manager.update_config(**{"wandb.entity_name": ""})

    def test_validation_disabled_strict_mode(self, clean_config_manager):
        """Test validation is disabled when strict_mode is False."""
        manager = clean_config_manager
        manager.load_config()

        # Disable strict mode
        manager.update_config(**{"validation.strict_mode": False})

        # Should not raise error even with empty project name
        manager.update_config(**{"wandb.project_name": ""})
        assert manager.get_value("wandb.project_name") == ""

    def test_validation_invalid_config_update(self, clean_config_manager):
        """Test validation of invalid configuration updates."""
        manager = clean_config_manager
        manager.load_config()

        # Test updating non-existent nested structure incorrectly
        with pytest.raises(ConfigurationError):
            manager.update_config(**{"invalid..nested...key": "value"})

    @patch("pathlib.Path.mkdir")
    def test_validation_path_creation_failure(self, mock_mkdir, clean_config_manager):
        """Test validation when path creation fails."""
        mock_mkdir.side_effect = PermissionError("Permission denied")

        manager = clean_config_manager

        with pytest.raises(ConfigurationError, match="Cannot create config directory"):
            manager.load_config()


class TestPublicAPI:
    """Test the public API functions."""

    def test_get_config_function(self, clean_config_manager):
        """Test get_config public function."""
        config = get_config()
        assert isinstance(config, DictConfig)
        assert config.wandb.project_name == "sleap-roots"

    def test_update_config_function(self, clean_config_manager):
        """Test update_config public function."""
        update_config(**{"wandb.project_name": "api-test"})

        config = get_config()
        assert config.wandb.project_name == "api-test"

    def test_get_value_function(self, clean_config_manager):
        """Test get_value public function."""
        value = get_value("wandb.project_name")
        assert value == "sleap-roots"

        default_value = get_value("nonexistent.key", "default")
        assert default_value == "default"

    def test_set_value_function(self, clean_config_manager):
        """Test set_value public function."""
        set_value("wandb.project_name", "api-set-test")

        value = get_value("wandb.project_name")
        assert value == "api-set-test"

    def test_callback_functions(self, clean_config_manager, mock_callback):
        """Test callback registration/unregistration functions."""
        register_config_callback(mock_callback)
        update_config(**{"wandb.project_name": "callback-api-test"})

        mock_callback.assert_called_once()

        mock_callback.reset_mock()
        unregister_config_callback(mock_callback)
        update_config(**{"wandb.project_name": "callback-api-test-2"})

        mock_callback.assert_not_called()

    def test_reset_config_function(self, clean_config_manager):
        """Test reset_config public function."""
        update_config(**{"wandb.project_name": "to-be-reset"})
        assert get_value("wandb.project_name") == "to-be-reset"

        reset_config()
        assert get_value("wandb.project_name") == "sleap-roots"

    def test_save_and_load_functions(self, clean_config_manager):
        """Test save_config and load_config_from_file functions."""
        update_config(**{"wandb.project_name": "save-api-test"})

        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
            temp_path = f.name

        try:
            save_config(temp_path)
            assert Path(temp_path).exists()

            reset_config()
            assert get_value("wandb.project_name") == "sleap-roots"

            loaded_config = load_config_from_file(temp_path)
            assert loaded_config.wandb.project_name == "save-api-test"

        finally:
            try:
                Path(temp_path).unlink()
            except FileNotFoundError:
                pass


class TestLegacyCompatibility:
    """Test backward compatibility with old CONFIG system."""

    def test_legacy_config_getitem(self, clean_config_manager):
        """Test LegacyConfig __getitem__ method."""
        legacy_config = LegacyConfig()

        project_name = legacy_config["project_name"]
        assert project_name == "sleap-roots"

    def test_legacy_config_setitem(self, clean_config_manager):
        """Test LegacyConfig __setitem__ method."""
        legacy_config = LegacyConfig()

        legacy_config["project_name"] = "legacy-test"
        assert legacy_config["project_name"] == "legacy-test"

        # Should also update the main config
        assert get_value("wandb.project_name") == "legacy-test"

    def test_legacy_config_get_method(self, clean_config_manager):
        """Test LegacyConfig get method."""
        legacy_config = LegacyConfig()

        value = legacy_config.get("project_name")
        assert value == "sleap-roots"

        default_value = legacy_config.get("nonexistent_key", "default")
        assert default_value == "default"

    def test_legacy_config_update_method(self, clean_config_manager):
        """Test LegacyConfig update method."""
        legacy_config = LegacyConfig()

        legacy_config.update(
            project_name="legacy-update-test", entity_name="legacy-entity"
        )

        assert legacy_config["project_name"] == "legacy-update-test"
        assert legacy_config["entity_name"] == "legacy-entity"

    def test_global_config_instance(self, clean_config_manager):
        """Test global CONFIG instance works like old system."""
        # Should work like a dictionary
        assert CONFIG["project_name"] == "sleap-roots"

        CONFIG["project_name"] = "global-test"
        assert CONFIG["project_name"] == "global-test"

        CONFIG.update(entity_name="global-entity")
        assert CONFIG["entity_name"] == "global-entity"

    def test_migration_from_old_config(self, clean_config_manager, old_config_file):
        """Test migrating from old YAML config file."""
        migrated_config = migrate_from_old_config(old_config_file)

        assert migrated_config.wandb.project_name == "old-project"
        assert migrated_config.wandb.entity_name == "old-entity"
        assert migrated_config.wandb.experiment_name == "old-experiment"
        assert migrated_config.wandb.registry == "old-registry"

    def test_migration_file_not_found(self, clean_config_manager):
        """Test migration fails gracefully when file not found."""
        with pytest.raises(FileNotFoundError):
            migrate_from_old_config("/nonexistent/config.yaml")


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_concurrent_config_updates(self, clean_config_manager):
        """Test concurrent configuration updates are thread-safe."""
        manager = clean_config_manager
        manager.load_config()

        results = []

        def update_config_worker(worker_id):
            try:
                manager.update_config(
                    **{f"custom.worker_{worker_id}": f"value_{worker_id}"}
                )
                results.append(f"worker_{worker_id}_success")
            except Exception as e:
                results.append(f"worker_{worker_id}_error: {e}")

        # Run 10 concurrent updates
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(update_config_worker, i) for i in range(10)]
            concurrent.futures.wait(futures)

        # All updates should succeed
        assert len(results) == 10
        assert all("success" in result for result in results)

        # All values should be present
        config = manager.get_config()
        for i in range(10):
            assert config.custom[f"worker_{i}"] == f"value_{i}"

    def test_memory_usage_large_config(self, clean_config_manager):
        """Test memory usage with large configuration."""
        manager = clean_config_manager
        manager.load_config()

        # Create large config update
        large_update = {}
        for i in range(1000):
            large_update[f"custom.large_key_{i}"] = f"large_value_{i}" * 100

        manager.update_config(**large_update)

        # Verify config was updated
        config = manager.get_config()
        assert len(config.custom) >= 1000
        assert config.custom.large_key_0.startswith("large_value_0")

    def test_config_with_special_characters(self, clean_config_manager):
        """Test configuration with special characters and unicode."""
        manager = clean_config_manager
        manager.load_config()

        special_values = {
            "custom.unicode": "测试配置 🚀",
            "custom.special_chars": "!@#$%^&*()_+-=[]{}|;:,.<>?",
            "custom.newlines": "line1\nline2\nline3",
            "custom.quotes": 'single "double" quotes',
        }

        manager.update_config(**special_values)

        config = manager.get_config()
        assert config.custom.unicode == "测试配置 🚀"
        assert config.custom.special_chars == "!@#$%^&*()_+-=[]{}|;:,.<>?"
        assert "line1\nline2\nline3" in config.custom.newlines

    def test_config_type_coercion(self, clean_config_manager):
        """Test configuration type coercion and validation."""
        manager = clean_config_manager
        manager.load_config()

        # Test various types
        type_updates = {
            "custom.int_value": 42,
            "custom.float_value": 3.14,
            "custom.bool_value": True,
            "custom.list_value": [1, 2, 3],
            "custom.dict_value": {"nested": "value"},
        }

        manager.update_config(**type_updates)

        config = manager.get_config()
        assert config.custom.int_value == 42
        assert config.custom.float_value == 3.14
        assert config.custom.bool_value is True
        assert config.custom.list_value == [1, 2, 3]
        assert config.custom.dict_value.nested == "value"

    def test_config_deep_nesting(self, clean_config_manager):
        """Test deeply nested configuration structures."""
        manager = clean_config_manager
        manager.load_config()

        deep_key = "custom.level1.level2.level3.level4.level5.deep_value"
        manager.update_config(**{deep_key: "deeply_nested"})

        value = manager.get_value(deep_key)
        assert value == "deeply_nested"

    def test_invalid_key_paths(self, clean_config_manager):
        """Test behavior with invalid key paths."""
        manager = clean_config_manager
        manager.load_config()

        # Test empty key
        with pytest.raises(ConfigurationError):
            manager.update_config(**{"": "empty_key"})

        # Test key with only dots
        with pytest.raises(ConfigurationError):
            manager.update_config(**{"...": "dots_only"})

    def test_callback_registration_edge_cases(self, clean_config_manager):
        """Test edge cases in callback registration."""
        manager = clean_config_manager
        manager.load_config()

        callback = MagicMock()

        # Register same callback multiple times
        manager.register_callback(callback)
        manager.register_callback(callback)  # Should not duplicate

        manager.update_config(**{"wandb.project_name": "callback-edge-test"})

        # Should only be called once despite double registration
        callback.assert_called_once()

        # Unregister non-existent callback should not error
        fake_callback = MagicMock()
        manager.unregister_callback(fake_callback)  # Should not raise

    def test_file_io_edge_cases(self, clean_config_manager):
        """Test file I/O edge cases."""
        manager = clean_config_manager
        manager.load_config()

        # Test saving to read-only directory (should fail gracefully)
        with pytest.raises(ConfigurationError):
            manager.save_config("/root/readonly_config.yaml")

        # Test loading from non-existent file
        with pytest.raises(ConfigurationError):
            manager.load_from_file("/nonexistent/config.yaml")

        # Test loading from invalid YAML
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("invalid: yaml: content: [")
            invalid_path = f.name

        try:
            with pytest.raises(ConfigurationError):
                manager.load_from_file(invalid_path)
        finally:
            try:
                Path(invalid_path).unlink()
            except FileNotFoundError:
                pass

    def test_hydra_integration_edge_cases(self, clean_config_manager):
        """Test Hydra integration edge cases."""
        manager = clean_config_manager

        # Test loading with overrides
        config = manager.load_config(overrides=["wandb.project_name=override-test"])
        assert config.wandb.project_name == "override-test"

        # Test loading with invalid overrides
        with pytest.raises(ConfigurationError):
            manager.load_config(overrides=["invalid_override_syntax"])


class TestPerformance:
    """Test performance characteristics of the config system."""

    def test_config_access_performance(self, clean_config_manager):
        """Test performance of config access operations."""
        manager = clean_config_manager
        manager.load_config()

        # Time config access
        start_time = time.time()

        for _ in range(1000):
            _ = manager.get_value("wandb.project_name")

        access_time = time.time() - start_time

        # Should be fast (less than 1 second for 1000 accesses)
        assert access_time < 1.0

    def test_config_update_performance(self, clean_config_manager):
        """Test performance of config updates."""
        manager = clean_config_manager
        manager.load_config()

        # Time config updates
        start_time = time.time()

        for i in range(100):
            manager.update_config(**{f"custom.perf_test_{i}": f"value_{i}"})

        update_time = time.time() - start_time

        # Should be reasonably fast (less than 5 seconds for 100 updates)
        assert update_time < 5.0

    def test_callback_performance(self, clean_config_manager):
        """Test performance with many callbacks registered."""
        manager = clean_config_manager
        manager.load_config()

        # Register many callbacks
        callbacks = [MagicMock() for _ in range(100)]
        for callback in callbacks:
            manager.register_callback(callback)

        # Time a config update with all callbacks
        start_time = time.time()
        manager.update_config(**{"wandb.project_name": "performance-test"})
        callback_time = time.time() - start_time

        # Should complete in reasonable time (less than 1 second)
        assert callback_time < 1.0

        # All callbacks should have been called
        for callback in callbacks:
            callback.assert_called_once()


# Integration tests with real file operations
class TestIntegration:
    """Integration tests with real file system operations."""

    def test_full_config_lifecycle(self, clean_config_manager):
        """Test complete configuration lifecycle."""
        manager = clean_config_manager

        # 1. Load default config
        config = manager.load_config()
        assert config.wandb.project_name == "sleap-roots"

        # 2. Update some values
        manager.update_config(
            **{
                "wandb.project_name": "lifecycle-test",
                "wandb.experiment_name": "integration-test",
                "custom.test_flag": True,
            }
        )

        # 3. Save to file
        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
            save_path = f.name

        try:
            manager.save_config(save_path)

            # 4. Reset and reload from file
            manager.reset_to_defaults()
            assert manager.get_value("wandb.project_name") == "sleap-roots"

            loaded_config = manager.load_from_file(save_path)
            assert loaded_config.wandb.project_name == "lifecycle-test"
            assert loaded_config.wandb.experiment_name == "integration-test"
            assert loaded_config.custom.test_flag is True

        finally:
            try:
                Path(save_path).unlink()
            except FileNotFoundError:
                pass

    def test_legacy_migration_integration(self, clean_config_manager):
        """Test complete legacy migration workflow."""
        # Create old config file
        old_config = {
            "project_name": "legacy-integration",
            "entity_name": "legacy-entity-integration",
            "experiment_name": "legacy-experiment",
            "registry": "legacy-registry",
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.safe_dump(old_config, f)
            old_path = f.name

        try:
            # Migrate from old config
            migrated_config = migrate_from_old_config(old_path)

            # Verify migration worked
            assert migrated_config.wandb.project_name == "legacy-integration"
            assert migrated_config.wandb.entity_name == "legacy-entity-integration"

            # Test legacy interface still works
            assert CONFIG["project_name"] == "legacy-integration"
            assert CONFIG["entity_name"] == "legacy-entity-integration"

            # Test updates through legacy interface
            CONFIG["project_name"] = "updated-through-legacy"
            assert get_value("wandb.project_name") == "updated-through-legacy"

        finally:
            try:
                Path(old_path).unlink()
            except FileNotFoundError:
                pass
