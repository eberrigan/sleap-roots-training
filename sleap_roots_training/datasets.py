import wandb
import logging

from pathlib import Path
from typing import List, Optional

from sleap_roots_training.config import CONFIG

# Set up logging
logging.basicConfig(level=logging.INFO)

def make_dataset_artifact(artifact_name: str, 
                          dataset_path: str, 
                          link_to_registry: bool = False, 
                          description: Optional[str] = None, 
                          tags: Optional[List[str]] = None) -> wandb.Artifact:
    """Create a dataset artifact from the training data.
    
    Args:
        artifact_name: The name of the artifact to create.
        dataset_path: The path to the dataset. This should be a .slp file.
        link_to_registry: Whether to link the artifact to the registry.
        description: A description of the artifact.
        tags: A list of tags for the artifact.
    
    Returns:
        The created dataset artifact.
    """
    # Load the configuration
    PROJECT_NAME = CONFIG["project_name"]
    ENTITY_NAME = CONFIG["entity_name"]
    EXPERIMENT_NAME = CONFIG["experiment_name"]
    REGISTRY = CONFIG["registry"]
    COLLECTION_NAME = CONFIG["collection_name"]

    # Initialize the W&B run
    run = wandb.init(
        project=PROJECT_NAME,
        entity=ENTITY_NAME,
        job_type="build_dataset",
        name=EXPERIMENT_NAME,
        save_code=True
    )

    try:
        dataset_path = Path(dataset_path)
        artifact = wandb.Artifact(
            name=artifact_name,
            type="dataset",
            description=description if description else ""
        )

        # Add metadata
        artifact.metadata["data_path"] = dataset_path.as_posix()
        if tags:
            for tag in tags:
                artifact.metadata[tag] = True

        # Add the dataset file to the artifact
        artifact.add_file(local_path=dataset_path.as_posix(), overwrite=False)
        logging.info(f"Dataset artifact created: {artifact_name} from {dataset_path.as_posix()}.")

        # Log the artifact to the W&B run
        run.log_artifact(artifact, tags=tags)

        # Link the artifact to the registry if specified
        if link_to_registry:
            target_path = f"{ENTITY_NAME}-org/wandb-registry-{REGISTRY}/{COLLECTION_NAME}"
            logging.info(f"Linking {artifact_name} to registry {target_path}.")
            run.link_artifact(artifact=artifact, target_path=target_path)

        return artifact

    except Exception as e:
        logging.error(f"Error creating dataset artifact: {e}")
        raise e
    
    finally:
        # Always finish the run, even if an error occurs
        run.finish()
        logging.info("W&B run finished successfully.")
    



