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


def _referenced_paths(backend) -> List[str]:
    """Return the external file path(s) a video backend references, if any.

    SLEAP's ``search_paths`` relocation only rewrites a backend's singular
    ``filename`` (the current image), not the plural ``filenames`` list. For
    image-sequence backends we rebase each ``filenames`` entry onto the
    (possibly relocated) directory of ``filename`` so existence checks reflect
    ``search_paths``.
    """
    filenames = getattr(backend, "filenames", None)
    if filenames:
        current = getattr(backend, "filename", None)
        if current:
            reloc_dir = os.path.dirname(str(current))
            return [
                os.path.join(reloc_dir, os.path.basename(str(f))) for f in filenames
            ]
        return [str(f) for f in filenames]
    filename = getattr(backend, "filename", None)
    return [str(filename)] if filename else []


def inspect_package(
    path: str, search_paths: Optional[List[str]] = None
) -> Dict[str, Any]:
    """Inspect a SLEAP package's embedding status and recoverability.

    Args:
        path: Path to a ``.slp``/``.pkg.slp`` file.
        search_paths: Optional directories forwarded to ``sleap.load_file`` to relocate
            missing referenced videos by basename before checking their existence.

    Returns:
        A dict describing embedding status, per-video detail, and how (if at all) a
        broken package could be repaired from its own referenced videos:
        ``recoverable_via`` is ``"already_ok"`` (fully embedded), ``"referenced_videos"``
        (re-embeddable from reachable source files), or ``"none"``.
    """
    _sleap = _get_sleap()

    result: Dict[str, Any] = {
        "embedded": False,
        "loadable": False,
        "error": None,
        "n_user_frames": 0,
        "n_videos": 0,
        "n_videos_missing_pixels": 0,
        "recoverable_via": "none",
        "referenced_paths": [],
        "videos": [],
    }

    try:
        if search_paths:
            labels = _sleap.load_file(path, search_paths=search_paths)
        else:
            labels = _sleap.load_file(path)
    except Exception as e:
        result["error"] = f"{type(e).__name__}: {e}"
        return result

    result["loadable"] = True
    videos_with_user_frames = {
        id(lf.video) for lf in labels.labeled_frames if lf.has_user_instances
    }
    result["n_user_frames"] = sum(
        1 for lf in labels.labeled_frames if lf.has_user_instances
    )
    result["n_videos"] = len(labels.videos)

    missing_referenced: List[str] = []
    all_missing_reachable = True
    n_missing = 0

    for video in labels.videos:
        backend = video.backend
        has_user = id(video) in videos_with_user_frames
        embedded = _video_has_embedded(backend)
        refs = _referenced_paths(backend)
        refs_exist = all(os.path.exists(p) for p in refs) if refs else False
        result["videos"].append(
            {
                "backend_type": type(backend).__name__,
                "embedded": embedded,
                "has_user_frames": has_user,
                "referenced_paths": refs,
                "referenced_exists": refs_exist,
            }
        )
        if has_user and not embedded:
            n_missing += 1
            missing_referenced.extend(refs)
            if not refs_exist:
                all_missing_reachable = False

    result["n_videos_missing_pixels"] = n_missing
    result["referenced_paths"] = missing_referenced
    result["embedded"] = result["n_user_frames"] > 0 and n_missing == 0

    if result["embedded"]:
        result["recoverable_via"] = "already_ok"
    elif n_missing > 0 and all_missing_reachable:
        result["recoverable_via"] = "referenced_videos"
    else:
        result["recoverable_via"] = "none"

    return result


def _classify_recoverability(info: Dict[str, Any], data_path_embedded: bool) -> str:
    """Combine per-file inspection with the registry ``data_path`` tier-1 check.

    Returns one of ``already_ok`` (artifact already embedded), ``already_embedded``
    (the metadata data_path file on disk is embedded), ``referenced_videos``
    (re-embeddable from reachable source videos), or ``none``.
    """
    if info.get("embedded"):
        return "already_ok"
    if data_path_embedded:
        return "already_embedded"
    if info.get("recoverable_via") == "referenced_videos":
        return "referenced_videos"
    return "none"


def _latest_version(versions: List) -> Optional[object]:
    """Return the version tagged ``latest``, else the first, else None."""
    for v in versions:
        if "latest" in (getattr(v, "aliases", None) or []):
            return v
    return versions[0] if versions else None


def _find_slp(directory: str) -> Optional[str]:
    """Return the first ``.slp``/``.pkg.slp`` file under ``directory`` (sorted)."""
    import glob

    matches = sorted(glob.glob(os.path.join(directory, "**", "*.slp"), recursive=True))
    return matches[0] if matches else None


def make_dataset_artifact(
    artifact_name: str,
    dataset_path: str,
    link_to_registry: bool = False,
    description: Optional[str] = None,
    tags: Optional[List[str]] = None,
    require_embedded_images: bool = True,
    metadata: Optional[Dict[str, Any]] = None,
) -> wandb.Artifact:
    """Create a dataset artifact from the training data.

    Args:
        artifact_name: The name of the artifact to create.
        dataset_path: The path to the dataset. This should be a .slp file.
        link_to_registry: Whether to link the artifact to the registry.
        description: A description of the artifact.
        tags: A list of tags for the artifact.
        require_embedded_images: Whether to refuse (raise) when the package at
            `dataset_path` lacks embedded images. If False, a warning is logged and
            registration proceeds anyway.
        metadata: Additional metadata to merge into the artifact's metadata dict.

    Returns:
        The created dataset artifact.
    """
    # Load the configuration
    PROJECT_NAME = CONFIG["project_name"]
    ENTITY_NAME = CONFIG["entity_name"]
    EXPERIMENT_NAME = CONFIG["experiment_name"]
    REGISTRY = CONFIG["registry"]
    COLLECTION_NAME = CONFIG["collection_name"]

    # Guardrail: refuse to register a package that lacks embedded images, since it
    # cannot be used for remote training. Runs before wandb.init to avoid orphan runs.
    dataset_path = Path(dataset_path)
    if not has_embedded_images(dataset_path.as_posix()):
        message = (
            f"Refusing to register '{dataset_path.as_posix()}': it has no embedded "
            "images (or could not be read as a SLEAP package). Save it with "
            "`labels.save(path, with_images=True)`, or pass "
            "`require_embedded_images=False` to register anyway."
        )
        if require_embedded_images:
            raise ValueError(message)
        logging.warning(message)

    # Initialize the W&B run
    run = wandb.init(
        project=PROJECT_NAME,
        entity=ENTITY_NAME,
        job_type="build_dataset",
        name=EXPERIMENT_NAME,
        save_code=True,
    )

    try:
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
        if metadata:
            artifact.metadata.update(metadata)

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
