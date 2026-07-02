import os
import wandb
import logging

from pathlib import Path
from typing import Any, Dict, List, Optional

from sleap_roots_training.config import CONFIG

# sleap is heavy and optional at import time (the cross-platform test-imports job has no
# sleap). Keep it out of module import; _get_sleap() imports it lazily on first use and
# caches it here. Tests patch `sleap_roots_training.datasets.sleap` directly.
sleap = None

# Set up logging
logging.basicConfig(level=logging.INFO)


def _get_sleap():
    """Import and cache the ``sleap`` module lazily (kept out of module import time)."""
    global sleap
    if sleap is None:
        import sleap as _sleap

        sleap = _sleap
    return sleap


def _video_has_embedded(backend) -> bool:
    """Return True iff a video backend is an HDF5Video carrying embedded images.

    Never raises: any error accessing the backend (e.g. an unreadable source video)
    is treated as "not embedded".
    """
    try:
        return type(backend).__name__ == "HDF5Video" and bool(
            backend.has_embedded_images
        )
    except Exception:
        return False


def has_embedded_images(path: str) -> bool:
    """Return True iff the SLEAP package at ``path`` is trainable on its own.

    A package is trainable when it has at least one user-labeled frame and every video
    that carries user-labeled frames has embedded image data. Videos without user labels
    are ignored (packages may reference thousands of unused videos). Returns False if the
    file cannot be read as a SLEAP package.

    Args:
        path: Path to a ``.slp``/``.pkg.slp`` file.

    Returns:
        True if every user-labeled-frame video has embedded images, else False.
    """
    _sleap = _get_sleap()
    try:
        labels = _sleap.load_file(path)
    except Exception as e:
        logging.debug(f"has_embedded_images: could not load {path}: {e}")
        return False

    videos_with_user_frames = {
        id(lf.video) for lf in labels.labeled_frames if lf.has_user_instances
    }
    if not videos_with_user_frames:
        return False

    for video in labels.videos:
        if id(video) not in videos_with_user_frames:
            continue
        if not _video_has_embedded(video.backend):
            return False
    return True


def make_dataset_artifact(
    artifact_name: str,
    dataset_path: str,
    link_to_registry: bool = False,
    description: Optional[str] = None,
    tags: Optional[List[str]] = None,
) -> wandb.Artifact:
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
        save_code=True,
    )

    try:
        dataset_path = Path(dataset_path)
        artifact = wandb.Artifact(
            name=artifact_name,
            type="dataset",
            description=description if description else "",
        )

        # Add metadata
        artifact.metadata["data_path"] = dataset_path.as_posix()
        if tags:
            for tag in tags:
                artifact.metadata[tag] = True

        # Add the dataset file to the artifact
        artifact.add_file(local_path=dataset_path.as_posix(), overwrite=False)
        logging.info(
            f"Dataset artifact created: {artifact_name} from {dataset_path.as_posix()}."
        )

        # Log the artifact to the W&B run
        run.log_artifact(artifact, tags=tags)

        # Link the artifact to the registry if specified
        if link_to_registry:
            target_path = (
                f"{ENTITY_NAME}-org/wandb-registry-{REGISTRY}/{COLLECTION_NAME}"
            )
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
