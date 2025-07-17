import wandb
import logging
import re

from pathlib import Path
from typing import List, Optional

from sleap_roots_training.config import CONFIG

# Set up logging
logging.basicConfig(level=logging.INFO)


def validate_tags(tags: List[str]):
    """Validates tags before logging them to W&B.

    Args:
        tags (List[str]): List of tags to validate.
    """
    if not isinstance(tags, list):
        raise ValueError("Tags should be a list of strings.")
    for tag in tags:
        if not isinstance(tag, str):
            raise ValueError(f"Tag '{tag}' is not a string.")
        if not re.match(r"^[\w\- ]+$", tag):
            raise ValueError(
                f"Invalid W&B tag: '{tag}' — tags must only contain "
                "alphanumeric characters, hyphens (-), underscores (_), or spaces."
            )


def fetch_model_artifact_from_experiment(
    project_name, entity_name, artifact_name, wandb_version=None
):
    """Fetches a specific version of a model artifact from a W&B experiment.

    Args:
        project_name (str): Name of the W&B project.
        entity_name (str): Name of the W&B entity.
        artifact_name (str): Name of the artifact to fetch.
        wandb_version (str, optional): Specific version from the training run names to fetch. Defaults to latest.

    Returns:
        wandb.Artifact: The fetched artifact.
    """
    run = wandb.init(
        project=project_name, entity=entity_name, job_type="fetch_artifact"
    )
    artifact_version = f"{wandb_version}" if wandb_version else "latest"
    full_artifact_name = f"{artifact_name}:{artifact_version}"
    print(f"Fetching artifact '{full_artifact_name}' from project '{project_name}'.")
    artifact = run.use_artifact(f"{full_artifact_name}")
    print(f"Fetched artifact '{full_artifact_name}'.")
    artifact_dir = artifact.download()
    print(
        f"Fetched artifact '{artifact_name}:{artifact_version}' to directory '{artifact_dir}'."
    )
    run.finish()
    return artifact


def fetch_model_artifact_and_link_to_registry(
    project_name,
    entity_name,
    artifact_name,
    registry_name,
    collection_name,
    wandb_version=None,
):
    """Fetchs a specific version of a model artifact from a W&B experiment and links it to the registry.

    Args:
        project_name (str): Name of the W&B project.
        entity_name (str): Name of the W&B entity.
        artifact_name (str): Name of the artifact to fetch.
        registry_name (str): Name of the registry to link the artifact to.
        collection_name (str): Name of the collection to store the model artifact.
        wandb_version (str, optional): Specific version from the training run names to fetch. Defaults to latest.
    """
    run = wandb.init(
        project=project_name, entity=entity_name, job_type="fetch_artifact"
    )
    artifact_version = f"{wandb_version}" if wandb_version else "latest"
    full_artifact_name = f"{artifact_name}:{artifact_version}"
    print(f"Fetching artifact '{full_artifact_name}' from project '{project_name}'.")
    artifact = run.use_artifact(f"{full_artifact_name}")
    print(f"Fetched artifact '{full_artifact_name}'.")

    # Link the artifact to the registry
    full_registry_name = (
        f"{entity_name}-org/wandb-registry-{registry_name}/{collection_name}"
    )
    print(
        f"Linking artifact '{full_artifact_name}' to registry '{full_registry_name}'."
    )
    run.link_artifact(artifact, full_registry_name)
    print(
        f"Linked artifact '{artifact_name}:{artifact_version}' to registry '{full_registry_name}'."
    )
    run.finish()


def promote_model_in_registry(
    project_name, entity_name, registry_name, artifact_name, stage, wandb_version=None
):
    """Promotes a specific artifact in the W&B model registry to a given stage.

    Args:
        project_name (str): Name of the W&B project.
        registry_name (str): Name of the model registry.
        artifact_name (str): Name of the artifact to promote.
        stage (str): Stage to promote the artifact to (e.g., 'production', 'staging').

    Returns:
        None
    """
    run = wandb.init(
        project=project_name, entity=entity_name, job_type="promote_registry_artifact"
    )
    artifact_version = f"{wandb_version}" if wandb_version else "latest"
    full_artifact_name = f"{artifact_name}:{artifact_version}"
    artifact = run.use_artifact(f"{registry_name}/{full_artifact_name}:latest")
    artifact.aliases.append(stage)
    artifact.save()
    print(
        f"Promoted artifact '{full_artifact_name}' in registry '{registry_name}' to stage '{stage}'."
    )
    run.finish()
