"""
Configuration Migration Tool

This script helps users migrate from the old YAML-based config system 
to the new OmegaConf/Hydra-based dynamic configuration system.

Usage:
    python -m sleap_roots_training.migrate_config [--backup] [--verify]
    
Example:
    python -m sleap_roots_training.migrate_config --backup --verify
"""

import argparse
import logging
import shutil
import sys
from pathlib import Path
from typing import Dict, Any, Optional

# Try to import the new config system
try:
    from sleap_roots_training.config_new import (
        migrate_from_old_config,
        get_config,
        save_config,
        CONFIG as NEW_CONFIG,
    )

    NEW_CONFIG_AVAILABLE = True
except ImportError as e:
    NEW_CONFIG_AVAILABLE = False
    NEW_CONFIG_ERROR = str(e)

# Try to import the old config system
try:
    from sleap_roots_training.config import CONFIG_PATH as OLD_CONFIG_PATH

    OLD_CONFIG_AVAILABLE = True
except ImportError:
    OLD_CONFIG_AVAILABLE = False


logger = logging.getLogger(__name__)


class MigrationError(Exception):
    """Raised when migration encounters an error."""

    pass


def check_prerequisites() -> None:
    """
    Check that prerequisites for migration are met.

    Raises:
        MigrationError: If prerequisites are not met
    """
    if not NEW_CONFIG_AVAILABLE:
        raise MigrationError(
            f"New config system not available. Error: {NEW_CONFIG_ERROR}\n"
            "Please install required dependencies: pip install hydra-core omegaconf"
        )

    if not OLD_CONFIG_AVAILABLE:
        logger.warning(
            "Old config system not found. This may be expected if already migrated."
        )


def find_old_config_file() -> Optional[Path]:
    """
    Find the old config.yaml file.

    Returns:
        Path to old config file or None if not found
    """
    # Check the standard location
    if OLD_CONFIG_AVAILABLE and OLD_CONFIG_PATH.exists():
        return OLD_CONFIG_PATH

    # Check other common locations
    possible_locations = [
        Path.cwd() / "config.yaml",
        Path.home() / ".sleap_roots_training" / "config.yaml",
        Path(__file__).parent / "config.yaml",
    ]

    for location in possible_locations:
        if location.exists():
            logger.info(f"Found old config file at: {location}")
            return location

    return None


def backup_old_config(old_config_path: Path) -> Path:
    """
    Create a backup of the old configuration file.

    Args:
        old_config_path: Path to the old config file

    Returns:
        Path to the backup file
    """
    backup_path = old_config_path.with_suffix(".yaml.backup")
    counter = 1

    # Find available backup name
    while backup_path.exists():
        backup_path = old_config_path.with_suffix(f".yaml.backup.{counter}")
        counter += 1

    shutil.copy2(old_config_path, backup_path)
    logger.info(f"Created backup at: {backup_path}")
    return backup_path


def analyze_old_config(old_config_path: Path) -> Dict[str, Any]:
    """
    Analyze the old configuration file.

    Args:
        old_config_path: Path to old config file

    Returns:
        Analysis results
    """
    import yaml

    with open(old_config_path, "r") as f:
        old_config = yaml.safe_load(f)

    analysis = {
        "total_keys": len(old_config) if old_config else 0,
        "required_keys": [],
        "optional_keys": [],
        "unknown_keys": [],
        "config_data": old_config or {},
    }

    # Categorize keys
    known_keys = {
        "project_name",
        "entity_name",
        "experiment_name",
        "registry",
        "collection_name",
        "job_type",
    }

    if old_config:
        for key in old_config.keys():
            if key in known_keys:
                if old_config[key]:  # Has a value
                    analysis["required_keys"].append(key)
                else:
                    analysis["optional_keys"].append(key)
            else:
                analysis["unknown_keys"].append(key)

    return analysis


def perform_migration(old_config_path: Path, analysis: Dict[str, Any]) -> None:
    """
    Perform the actual migration.

    Args:
        old_config_path: Path to old config file
        analysis: Configuration analysis results
    """
    logger.info("Starting configuration migration...")

    try:
        # Migrate using the new config system
        migrated_config = migrate_from_old_config(old_config_path)

        logger.info("Migration completed successfully!")

        # Show what was migrated
        logger.info("Migrated settings:")
        for key in analysis["required_keys"]:
            old_value = analysis["config_data"][key]
            logger.info(f"  {key}: {old_value}")

        # Warn about unknown keys
        if analysis["unknown_keys"]:
            logger.warning("Unknown keys found in old config (not migrated):")
            for key in analysis["unknown_keys"]:
                logger.warning(f"  {key}: {analysis['config_data'][key]}")

    except Exception as e:
        raise MigrationError(f"Migration failed: {e}") from e


def verify_migration(old_config_path: Path, analysis: Dict[str, Any]) -> None:
    """
    Verify that migration was successful.

    Args:
        old_config_path: Path to old config file
        analysis: Configuration analysis results
    """
    logger.info("Verifying migration...")

    try:
        # Get the new config
        new_config = get_config()

        # Verify key mappings
        key_mappings = {
            "project_name": "wandb.project_name",
            "entity_name": "wandb.entity_name",
            "experiment_name": "wandb.experiment_name",
            "registry": "wandb.registry",
            "collection_name": "wandb.collection_name",
            "job_type": "wandb.job_type",
        }

        verification_passed = True

        for old_key in analysis["required_keys"]:
            if old_key in key_mappings:
                new_key_path = key_mappings[old_key].split(".")
                new_value = new_config

                # Navigate to the nested value
                for part in new_key_path:
                    new_value = getattr(new_value, part, None)
                    if new_value is None:
                        break

                old_value = analysis["config_data"][old_key]

                if new_value == old_value:
                    logger.info(f"✓ {old_key}: {old_value} -> {new_value}")
                else:
                    logger.error(f"✗ {old_key}: {old_value} -> {new_value} (MISMATCH)")
                    verification_passed = False

        if verification_passed:
            logger.info("✓ Migration verification passed!")
        else:
            logger.error("✗ Migration verification failed!")
            raise MigrationError("Migration verification failed")

    except Exception as e:
        raise MigrationError(f"Migration verification failed: {e}") from e


def show_usage_examples() -> None:
    """Show examples of how to use the new config system."""
    logger.info("\n" + "=" * 60)
    logger.info("NEW CONFIG SYSTEM USAGE EXAMPLES")
    logger.info("=" * 60)

    examples = [
        (
            "Import the new config system:",
            "from sleap_roots_training.config_new import get_config, update_config",
        ),
        (
            "Get current configuration:",
            "config = get_config()\nprint(config.wandb.project_name)",
        ),
        (
            "Update configuration dynamically:",
            "update_config(**{'wandb.project_name': 'new-project'})",
        ),
        (
            "Access specific values:",
            "from sleap_roots_training.config_new import get_value\nproject = get_value('wandb.project_name')",
        ),
        (
            "Backward compatibility (legacy):",
            "from sleap_roots_training.config_new import CONFIG\nprint(CONFIG['project_name'])  # Still works!",
        ),
        (
            "Register for config changes:",
            "from sleap_roots_training.config_new import register_config_callback\n"
            "def on_config_change(config):\n    print(f'Config changed: {config.wandb.project_name}')\n"
            "register_config_callback(on_config_change)",
        ),
    ]

    for title, code in examples:
        logger.info(f"\n{title}")
        logger.info(f"  {code}")


def main():
    """Main migration function."""
    parser = argparse.ArgumentParser(
        description="Migrate from old config system to new OmegaConf/Hydra system"
    )
    parser.add_argument(
        "--backup", action="store_true", help="Create backup of old config file"
    )
    parser.add_argument(
        "--verify", action="store_true", help="Verify migration was successful"
    )
    parser.add_argument(
        "--examples",
        action="store_true",
        help="Show usage examples for new config system",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )
    parser.add_argument(
        "--config-path",
        type=str,
        help="Path to old config file (auto-detected if not provided)",
    )

    args = parser.parse_args()

    # Setup logging
    log_level = logging.INFO if args.verbose else logging.WARNING
    logging.basicConfig(level=log_level, format="%(levelname)s: %(message)s")

    try:
        # Check prerequisites
        check_prerequisites()

        # Find old config file
        if args.config_path:
            old_config_path = Path(args.config_path)
            if not old_config_path.exists():
                raise MigrationError(f"Config file not found: {old_config_path}")
        else:
            old_config_path = find_old_config_file()
            if not old_config_path:
                logger.info(
                    "No old config file found. You may already be using the new system!"
                )
                if args.examples:
                    show_usage_examples()
                return

        logger.info(f"Found old config file: {old_config_path}")

        # Analyze old config
        analysis = analyze_old_config(old_config_path)
        logger.info(f"Old config has {analysis['total_keys']} keys")

        # Create backup if requested
        backup_path = None
        if args.backup:
            backup_path = backup_old_config(old_config_path)

        # Perform migration
        perform_migration(old_config_path, analysis)

        # Verify migration if requested
        if args.verify:
            verify_migration(old_config_path, analysis)

        # Show usage examples if requested
        if args.examples:
            show_usage_examples()

        logger.info("\n" + "=" * 60)
        logger.info("MIGRATION COMPLETED SUCCESSFULLY!")
        logger.info("=" * 60)
        logger.info("You can now use the new dynamic configuration system.")
        logger.info("The old config file is no longer needed.")
        if backup_path:
            logger.info(f"A backup was created at: {backup_path}")
        logger.info("\nTo see usage examples, run:")
        logger.info("  python -m sleap_roots_training.migrate_config --examples")

    except MigrationError as e:
        logger.error(f"Migration failed: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        logger.info("Migration cancelled by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error during migration: {e}")
        if args.verbose:
            logger.exception("Full error details:")
        sys.exit(1)


if __name__ == "__main__":
    main()
