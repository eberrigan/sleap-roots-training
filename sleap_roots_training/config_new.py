"""
Modern configuration system using OmegaConf and Hydra.

This module provides a dynamic, type-safe configuration system that replaces
the static CONFIG dictionary approach. Key features:

- Type-safe structured configs using dataclasses
- Dynamic configuration updates without kernel restarts
- Runtime validation and error handling  
- Event system for config change notifications
- Hydra integration for composition and CLI overrides
- Backward compatibility with existing code
- Comprehensive testing with edge case handling

Example:
    >>> from sleap_roots_training.config_new import get_config, update_config
    >>> config = get_config()
    >>> print(config.project_name)
    >>> update_config(project_name="new-project")
    >>> # Config changes are immediately available everywhere
"""

import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union
from omegaconf import DictConfig, OmegaConf
from hydra import compose, initialize
from hydra.core.config_store import ConfigStore
from hydra.core.global_hydra import GlobalHydra


# Set up logging
logger = logging.getLogger(__name__)


@dataclass
class WandBConfig:
    """Configuration for Weights & Biases integration."""

    project_name: str = "sleap-roots"
    entity_name: str = "eberrigan-salk-institute-for-biological-studies"
    experiment_name: Optional[str] = None
    registry: Optional[str] = None
    collection_name: Optional[str] = None
    job_type: Optional[str] = None


@dataclass
class PathConfig:
    """Configuration for file paths and directories."""

    config_dir: str = "${oc.env:HOME}/.sleap_roots_training"
    data_dir: Optional[str] = None
    models_dir: Optional[str] = None
    logs_dir: Optional[str] = None


@dataclass
class ValidationConfig:
    """Configuration validation settings."""

    strict_mode: bool = True
    validate_paths: bool = True
    validate_wandb: bool = True
    allow_missing_optional: bool = True


@dataclass
class SleapRootsConfig:
    """Root configuration schema for SLEAP Roots Training."""

    wandb: WandBConfig = field(default_factory=WandBConfig)
    paths: PathConfig = field(default_factory=PathConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)

    # Additional fields for extensibility
    custom: Dict[str, Any] = field(default_factory=dict)
    version: str = "2.0.0"


class ConfigurationError(Exception):
    """Raised when configuration validation fails."""

    pass


class ConfigManager:
    """
    Thread-safe singleton configuration manager.

    Provides dynamic access to configuration with automatic validation,
    change notifications, and backward compatibility.
    """

    _instance: Optional["ConfigManager"] = None
    _lock = threading.RLock()

    def __new__(cls) -> "ConfigManager":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if hasattr(self, "_initialized"):
            return

        self._config: Optional[DictConfig] = None
        self._callbacks: List[Callable[[DictConfig], None]] = []
        self._initialized = True
        self._setup_hydra()

    def _setup_hydra(self):
        """Setup Hydra configuration store."""
        try:
            if GlobalHydra().is_initialized():
                GlobalHydra.instance().clear()

            cs = ConfigStore.instance()
            cs.store(name="base_config", node=SleapRootsConfig)

        except Exception as e:
            logger.warning(f"Failed to setup Hydra: {e}")

    def load_config(
        self,
        config_name: str = "base_config",
        config_path: Optional[str] = None,
        overrides: Optional[List[str]] = None,
    ) -> DictConfig:
        """
        Load configuration using Hydra.

        Args:
            config_name: Name of the config to load
            config_path: Path to config directory
            overrides: List of config overrides

        Returns:
            Loaded and validated configuration

        Raises:
            ConfigurationError: If config loading or validation fails
        """
        with self._lock:
            try:
                # Clear any existing Hydra instance
                if GlobalHydra().is_initialized():
                    GlobalHydra.instance().clear()

                # Initialize with config store
                with initialize(config_path=config_path, version_base=None):
                    cfg = compose(config_name=config_name, overrides=overrides or [])

                # Make config non-struct to allow dynamic updates
                OmegaConf.set_struct(cfg, False)
                self._config = cfg
                self._validate_config(cfg)
                self._notify_callbacks(cfg)

                logger.info(f"Configuration loaded successfully: {config_name}")
                return cfg

            except Exception as e:
                error_msg = f"Failed to load configuration: {e}"
                logger.error(error_msg)
                raise ConfigurationError(error_msg) from e

    def get_config(self) -> DictConfig:
        """
        Get current configuration.

        Returns:
            Current configuration

        Raises:
            ConfigurationError: If no configuration is loaded
        """
        if self._config is None:
            # Try to load default configuration
            try:
                return self.load_config()
            except Exception as e:
                raise ConfigurationError(
                    "No configuration loaded and failed to load default"
                ) from e

        return self._config

    def update_config(self, **kwargs) -> None:
        """
        Update configuration values dynamically.

        Args:
            **kwargs: Configuration key-value pairs to update

        Example:
            update_config(wandb.project_name="new-project", paths.data_dir="/new/path")
        """
        with self._lock:
            config_was_none = self._config is None
            if config_was_none:
                # Load config without triggering callbacks yet
                try:
                    from hydra.core.global_hydra import GlobalHydra

                    if GlobalHydra().is_initialized():
                        GlobalHydra.instance().clear()

                    # Initialize with config store
                    with initialize(config_path=None, version_base=None):
                        cfg = compose(config_name="base_config", overrides=[])

                    # Make config non-struct to allow dynamic updates
                    OmegaConf.set_struct(cfg, False)
                    self._config = cfg
                    self._validate_config(cfg)

                except Exception as e:
                    error_msg = f"Failed to load configuration: {e}"
                    logger.error(error_msg)
                    raise ConfigurationError(error_msg) from e

            # Update configuration using OmegaConf
            for key, value in kwargs.items():
                # Validate key format
                if not key or key.startswith(".") or key.endswith(".") or ".." in key:
                    raise ConfigurationError(f"Invalid config key format: '{key}'")

                try:
                    OmegaConf.update(self._config, key, value)
                except Exception as e:
                    logger.error(f"Failed to update config key '{key}': {e}")
                    raise ConfigurationError(
                        f"Invalid config update: {key}={value}"
                    ) from e

            # Validate after updates
            self._validate_config(self._config)
            self._notify_callbacks(self._config)

            logger.info(f"Configuration updated: {kwargs}")

    def get_value(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value by key path.

        Args:
            key: Dot-separated key path (e.g., "wandb.project_name")
            default: Default value if key not found

        Returns:
            Configuration value or default
        """
        try:
            config = self.get_config()
            return OmegaConf.select(config, key, default=default)
        except Exception:
            return default

    def set_value(self, key: str, value: Any) -> None:
        """
        Set a configuration value by key path.

        Args:
            key: Dot-separated key path
            value: Value to set
        """
        self.update_config(**{key: value})

    def register_callback(self, callback: Callable[[DictConfig], None]) -> None:
        """
        Register a callback for configuration changes.

        Args:
            callback: Function to call when config changes
        """
        with self._lock:
            if callback not in self._callbacks:
                self._callbacks.append(callback)
                callback_name = getattr(callback, "__name__", str(callback))
                logger.debug(f"Registered config callback: {callback_name}")

    def unregister_callback(self, callback: Callable[[DictConfig], None]) -> None:
        """
        Unregister a configuration change callback.

        Args:
            callback: Callback function to remove
        """
        with self._lock:
            if callback in self._callbacks:
                self._callbacks.remove(callback)
                callback_name = getattr(callback, "__name__", str(callback))
                logger.debug(f"Unregistered config callback: {callback_name}")

    def _validate_config(self, config: DictConfig) -> None:
        """
        Validate configuration values.

        Args:
            config: Configuration to validate

        Raises:
            ConfigurationError: If validation fails
        """
        if not config.validation.strict_mode:
            return

        try:
            # Validate W&B settings
            if config.validation.validate_wandb:
                self._validate_wandb_config(config.wandb)

            # Validate paths
            if config.validation.validate_paths:
                self._validate_path_config(config.paths)

        except Exception as e:
            raise ConfigurationError(f"Configuration validation failed: {e}") from e

    def _validate_wandb_config(self, wandb_config: DictConfig) -> None:
        """Validate W&B configuration."""
        if not wandb_config.project_name:
            raise ConfigurationError("wandb.project_name is required")

        if not wandb_config.entity_name:
            raise ConfigurationError("wandb.entity_name is required")

    def _validate_path_config(self, path_config: DictConfig) -> None:
        """Validate path configuration."""
        # Resolve the config directory path
        config_dir_str = OmegaConf.select(path_config, "config_dir")
        if config_dir_str:
            # Handle environment variable expansion
            import os

            config_dir_str = os.path.expandvars(config_dir_str)
            config_dir = Path(config_dir_str)

            # Create config directory if it doesn't exist
            try:
                config_dir.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                raise ConfigurationError(
                    f"Cannot create config directory {config_dir}: {e}"
                )

    def _notify_callbacks(self, config: DictConfig) -> None:
        """Notify all registered callbacks of config changes."""
        for callback in self._callbacks:
            try:
                callback(config)
            except Exception as e:
                logger.error(f"Config callback {callback.__name__} failed: {e}")

    def reset_to_defaults(self) -> None:
        """Reset configuration to default values."""
        with self._lock:
            self._config = None
            self.load_config()
            logger.info("Configuration reset to defaults")

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to plain dictionary.

        Returns:
            Configuration as dictionary
        """
        config = self.get_config()
        return OmegaConf.to_container(config, resolve=True)

    def save_config(self, path: Union[str, Path]) -> None:
        """
        Save current configuration to file.

        Args:
            path: Path to save configuration
        """
        config = self.get_config()
        path = Path(path)

        try:
            with open(path, "w") as f:
                OmegaConf.save(config, f)
            logger.info(f"Configuration saved to {path}")
        except Exception as e:
            raise ConfigurationError(f"Failed to save config to {path}: {e}") from e

    def load_from_file(self, path: Union[str, Path]) -> DictConfig:
        """
        Load configuration from file.

        Args:
            path: Path to configuration file

        Returns:
            Loaded configuration
        """
        path = Path(path)

        try:
            config = OmegaConf.load(path)
            self._config = config
            self._validate_config(config)
            self._notify_callbacks(config)
            logger.info(f"Configuration loaded from {path}")
            return config
        except Exception as e:
            raise ConfigurationError(f"Failed to load config from {path}: {e}") from e


# Global config manager instance
_config_manager = ConfigManager()


# Public API functions
def get_config() -> DictConfig:
    """Get current configuration."""
    return _config_manager.get_config()


def update_config(**kwargs) -> None:
    """Update configuration values."""
    _config_manager.update_config(**kwargs)


def get_value(key: str, default: Any = None) -> Any:
    """Get configuration value by key path."""
    return _config_manager.get_value(key, default)


def set_value(key: str, value: Any) -> None:
    """Set configuration value by key path."""
    _config_manager.set_value(key, value)


def register_config_callback(callback: Callable[[DictConfig], None]) -> None:
    """Register callback for configuration changes."""
    _config_manager.register_callback(callback)


def unregister_config_callback(callback: Callable[[DictConfig], None]) -> None:
    """Unregister configuration change callback."""
    _config_manager.unregister_callback(callback)


def reset_config() -> None:
    """Reset configuration to defaults."""
    _config_manager.reset_to_defaults()


def save_config(path: Union[str, Path]) -> None:
    """Save configuration to file."""
    _config_manager.save_config(path)


def load_config_from_file(path: Union[str, Path]) -> DictConfig:
    """Load configuration from file."""
    return _config_manager.load_from_file(path)


# Backward compatibility helpers
def migrate_from_old_config(old_config_path: Union[str, Path]) -> DictConfig:
    """
    Migrate from old YAML-based config system.

    Args:
        old_config_path: Path to old config.yaml file

    Returns:
        Migrated configuration
    """
    import yaml

    old_path = Path(old_config_path)
    if not old_path.exists():
        raise FileNotFoundError(f"Old config file not found: {old_path}")

    # Load old config
    with open(old_path, "r") as f:
        old_data = yaml.safe_load(f)

    # Map old keys to new structure
    new_config = {
        "wandb": {
            "project_name": old_data.get("project_name", "sleap-roots"),
            "entity_name": old_data.get("entity_name", ""),
            "experiment_name": old_data.get("experiment_name"),
            "registry": old_data.get("registry"),
            "collection_name": old_data.get("collection_name"),
            "job_type": old_data.get("job_type"),
        }
    }

    # Update current config
    update_config(
        **{f"wandb.{k}": v for k, v in new_config["wandb"].items() if v is not None}
    )

    logger.info(f"Migrated configuration from {old_path}")
    return get_config()


# Legacy compatibility - provide CONFIG-like interface
class LegacyConfig:
    """Backward compatibility wrapper for old CONFIG dictionary access."""

    def __getitem__(self, key: str) -> Any:
        """Get config value with dictionary-like access."""
        # Map old keys to new structure
        key_mapping = {
            "project_name": "wandb.project_name",
            "entity_name": "wandb.entity_name",
            "experiment_name": "wandb.experiment_name",
            "registry": "wandb.registry",
            "collection_name": "wandb.collection_name",
            "job_type": "wandb.job_type",
        }

        mapped_key = key_mapping.get(key, key)

        # Use a sentinel to detect if key doesn't exist
        sentinel = object()
        value = get_value(mapped_key, sentinel)

        if value is sentinel:
            raise KeyError(f"Config key '{key}' not found")

        return value

    def __setitem__(self, key: str, value: Any) -> None:
        """Set config value with dictionary-like access."""
        key_mapping = {
            "project_name": "wandb.project_name",
            "entity_name": "wandb.entity_name",
            "experiment_name": "wandb.experiment_name",
            "registry": "wandb.registry",
            "collection_name": "wandb.collection_name",
            "job_type": "wandb.job_type",
        }

        mapped_key = key_mapping.get(key, key)
        set_value(mapped_key, value)

    def get(self, key: str, default: Any = None) -> Any:
        """Get config value with default."""
        try:
            return self[key]
        except:
            return default

    def update(self, **kwargs) -> None:
        """Update multiple config values."""
        for key, value in kwargs.items():
            self[key] = value


# Global legacy compatibility instance
CONFIG = LegacyConfig()
