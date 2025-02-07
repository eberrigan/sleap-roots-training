import yaml
from pathlib import Path
from typing import Dict, Any, List

# Define the path to the config file
CONFIG_PATH = Path(__file__).parent / "config.yaml"

# Default configuration dictionary for a wandb run
DEFAULT_CONFIG: Dict[str, Any] = {
    "project_name": "sleap-roots",
    "entity_name": "eberrigan-salk-institute-for-biological-studies",
    "experiment_name": None,
    "registry": None,
    "collection_name": None,
    "job_type": None,
}


def create_default_config() -> None:
    """Creates a default config.yaml file if it does not exist."""
    if not CONFIG_PATH.exists():
        with open(CONFIG_PATH, "w") as file:
            yaml.safe_dump(DEFAULT_CONFIG, file)
        print(f"Default config.yaml created at {CONFIG_PATH}")


def load_config() -> Dict[str, Any]:
    """Loads configuration from the YAML file.

    Returns:
        Dict[str, Any]: A dictionary containing the configuration parameters.
    """
    if not CONFIG_PATH.exists():
        create_default_config()
    
    with open(CONFIG_PATH, "r") as file:
        return yaml.safe_load(file)


def save_config(updated_config: Dict[str, Any]) -> None:
    """Saves the updated configuration to the YAML file.

    Args:
        updated_config (Dict[str, Any]): The updated configuration dictionary.
    """
    with open(CONFIG_PATH, "w") as file:
        yaml.safe_dump(updated_config, file)
    print(f"Configuration updated successfully at {CONFIG_PATH}.")


def update_config(**kwargs: Any) -> None:
    """Updates specific configuration values and saves the changes.

    Args:
        **kwargs (Any): Key-value pairs of configuration parameters to update.

    Example:
        update_config(project_name="new_project", entity_name="new_entity")
    """
    global CONFIG
    CONFIG.update(kwargs)  # Update only provided keys
    save_config(CONFIG)
    print(f"CONFIG updated to {CONFIG}.")


def reset_config() -> None:
    """Resets the configuration to the default settings."""
    global CONFIG
    CONFIG = DEFAULT_CONFIG.copy()
    save_config(CONFIG)
    print("Configuration has been reset to default values.")


# Ensure the config file is present and load it
CONFIG = load_config() 

# Assign constants from the loaded config
PROJECT_NAME: str = CONFIG["project_name"]
ENTITY_NAME: str = CONFIG["entity_name"]
EXPERIMENT_NAME: str = CONFIG["experiment_name"]
REGISTRY: str = CONFIG["registry"]
COLLECTION_NAME: str = CONFIG["collection_name"]
