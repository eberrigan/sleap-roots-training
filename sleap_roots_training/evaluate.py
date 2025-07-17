import wandb
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sleap
import numpy as np
import datetime
import logging

from matplotlib.patches import ConnectionPatch
from pathlib import Path
from typing import List, Optional, Dict
from wandb.sdk.wandb_run import Run
from wandb.sdk.artifacts.artifact import Artifact

from sleap_roots_training.config import CONFIG

# Set up logging
logging.basicConfig(level=logging.INFO)


def create_artifact_name(group: str, version: str) -> str:
    """Return the artifact name for a given group and version.

    Example: sorghum-soybean-primary-2025-01-07_v001
    Syntax is {group}_v{version}

    Args:
        group (str): The group name.
        version (str): The version number.

    Returns:
        str: The artifact name.
    """
    return f"{group}_v{version}"


def fetch_model_artifact(
    run: Run, entity_name: str, registry: str, artifact_name: str, alias: str = "latest"
) -> wandb.Artifact:
    """Return wandb.Artifact object for a given model artifact.

    Args:
        run (wandb.Run): The W&B run object.
        entity_name (str): W&B entity name.
        registry (str): W&B registry name.
        artifact_name (str): Name of the model artifact.
        alias (str): Alias of the model artifact. Default is "latest".

    Returns:
        wandb.Artifact: The model artifact.
    """
    full_artifact_name = (
        f"{entity_name}-org/wandb-registry-{registry}/{artifact_name}:{alias}"
    )
    print(f"Fetching artifact: {full_artifact_name}")
    artifact = run.use_artifact(f"{full_artifact_name}")
    return artifact


def get_eval_metadata(
    artifact: wandb.Artifact, metadata_key: str = "dist_avg"
) -> float:
    """Return model artifact metric from metadata.

    Args:
        artifact (wandb.Artifact): The model artifact.
        metadata_key (str): The key of the metric to retrieve. Default is "dist_avg".

    Returns:
        float: The metric value.
    """
    metadata = artifact.metadata
    return metadata.get(metadata_key)


def get_predictions(filename: str, model_path: str, overwrite=False) -> sleap.Labels:
    """Get predictions for a given video file and model path.

    Args:
        filename (str): The path to the video file.
        model_path (str): The path to the model file.
        overwrite (bool): Whether to overwrite existing predictions. Default is False.

    Returns:
        sleap.Labels: The predictions object.
    """
    predictor = sleap.load_model(model_path, progress_reporting="none")
    predictor_name = Path(model_path).stem
    preds_path = f"{filename[:-3]}_{predictor_name}.predictions.slp"
    if Path(preds_path).exists() and not overwrite:
        return sleap.load_file(preds_path)
    else:
        video = sleap.load_video(filename, dataset="vol", channels_first=False)
        predictions = predictor.predict(video)
        predictions.save(preds_path)
        return predictions


def get_test_data(model_artifact: wandb.Artifact) -> sleap.Labels:
    """Return test data from a model artifact.

    Args:
        model_artifact (wandb.Artifact): The model artifact to fetch test data from.

    Returns:
        sleap.Labels: The test data labels.
    """
    # Get the manifest entry for the test data file
    config_entry = model_artifact.get_entry("training_config.json")

    # Download the file locally
    config_path = config_entry.download()

    # Load the file using SLEAP
    cfg = sleap.load_config(config_path)
    sleap_labels = sleap.load_file(cfg.data.labels.test_labels)
    print(f"Loaded test data from {cfg.data.labels.test_labels}.")
    return sleap_labels


def predictions_viz(
    output_dir: str,
    filename: str,
    groups: List[str],
    frame_idx: int = 1,
    model_version: str = "002",
    overwrite=False,
):
    """Function to fetch model artifacts, run predictions, and save visualizations.

    Args:
        output_dir (str): Path to save outputs.
        filename (str): Path to the video file for predictions.
        groups (List[str]): List of group names to fetch artifacts from.
        frame_idx (int): Frame index for visualization. Default is 1.
        model_version (str): Model version to fetch from the registry. Default is "002".
        overwrite (bool): Whether to overwrite existing predictions. Default is False.
    """
    PROJECT_NAME = CONFIG["project_name"]
    ENTITY_NAME = CONFIG["entity_name"]
    EXPERIMENT_NAME = CONFIG["experiment_name"]
    REGISTRY = CONFIG["registry"]

    # Initialize W&B run
    run = wandb.init(
        project=PROJECT_NAME,
        entity=ENTITY_NAME,
        name=EXPERIMENT_NAME,
        job_type="predictions_viz",
        group=EXPERIMENT_NAME,
    )

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    for group in groups:
        artifact_name = create_artifact_name(group, model_version)
        try:
            artifact = fetch_model_artifact(
                run=run,
                entity_name=ENTITY_NAME,
                registry=REGISTRY,
                artifact_name=artifact_name,
                alias="latest",
            )
            artifact_dir = artifact.download(skip_cache=False)
            print(f"Downloaded artifact: {artifact_name} to {artifact_dir}")
        except Exception as e:
            wandb.alert(
                f"Error fetching artifact: {artifact_name}. Exception: {str(e)}"
            )
            continue

        if artifact_dir:
            try:
                predictions = get_predictions(
                    filename, model_path=artifact_dir, overwrite=overwrite
                )
                labeled_frame = predictions[frame_idx]

                # Visualize predictions
                sleap.nn.viz.plot_img(labeled_frame.image, scale=1.0)
                sleap.nn.viz.plot_instances(labeled_frame.instances, lw=2, ms=50)
                plt.title(f"Predictions for {artifact_name} at frame {frame_idx}")
                plt.savefig(
                    Path(output_dir)
                    / f"{Path(filename).stem}_{artifact_name}_frame_{frame_idx}.png"
                )
                plt.close()  # Close the plot to avoid memory issues
            except Exception as e:
                wandb.alert(
                    f"Error processing artifact: {artifact_name}. Exception: {str(e)}"
                )

    run.finish()


def predictions_viz_multiple_files(
    output_dir: str,
    filenames: list,
    groups: List[str],
    tags: List[str],
    frame_idx: int = 1,
    model_version: str = "002",
    overwrite=False,
    figsize_scale: float = 5,
    dpi: int = 300,
    transparent: bool = False,
    font_settings: dict = None,
    colormap: str = "magma",
    spacing: dict = None,
) -> None:
    """Visualize predictions for multiple files and models in a grid with customizable settings.

    Args:
        output_dir (str): Path to save outputs.
        filenames (list): List of video file paths for predictions.
        groups (List[str]): List of group names to fetch model artifacts from.
        frame_idx (int): Frame index for visualization. Default is 1.
        model_version (str): Model version to fetch from the registry. Default is "002".
        overwrite (bool): Whether to overwrite existing predictions. Default is False.
        figsize_scale (float): Scale factor for figure size. Default is 5.
        dpi (int): Resolution for saved figure. Default is 300.
        transparent (bool): Whether to save figure with a transparent background. Default is False.
        font_settings (dict): Custom font settings (e.g., font sizes for labels, title).
        colormap (str): Colormap for visualization. Default is "magma".
        spacing (dict): Adjust subplot spacing (wspace, hspace, left, right, top, bottom).

    Returns:
        None
    """
    PROJECT_NAME = CONFIG["project_name"]
    ENTITY_NAME = CONFIG["entity_name"]
    EXPERIMENT_NAME = CONFIG["experiment_name"]
    REGISTRY = CONFIG["registry"]

    # Initialize W&B run
    run = wandb.init(
        project=PROJECT_NAME,
        entity=ENTITY_NAME,
        name=EXPERIMENT_NAME,
        job_type="predictions_viz_multiple",
        tags=tags,
        group=EXPERIMENT_NAME,
    )

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    num_files = len(filenames)
    num_models = len(groups)
    fig, axes = plt.subplots(
        num_files,
        num_models,
        figsize=(figsize_scale * num_models, figsize_scale * num_files),
    )

    # Set figure transparency if required
    if transparent:
        fig.patch.set_alpha(0)

    # Apply font settings if provided
    if font_settings:
        plt.rcParams.update(font_settings)

    # Set colormap
    plt.set_cmap(colormap)

    for file_idx, filename in enumerate(filenames):
        if not Path(filename).exists():
            print(f"File not found: {filename}")
            continue

        for model_idx, group in enumerate(groups):
            artifact_name = create_artifact_name(group, model_version)
            try:
                artifact = fetch_model_artifact(
                    run=run,
                    entity_name=ENTITY_NAME,
                    registry=REGISTRY,
                    artifact_name=artifact_name,
                    alias="latest",
                )
                artifact_dir = artifact.download(skip_cache=False)
                print(f"Downloaded artifact: {artifact_name} to {artifact_dir}")
            except Exception as e:
                print(f"Error fetching artifact: {artifact_name}. Exception: {e}")
                continue

            if artifact_dir:
                try:
                    predictions = get_predictions(
                        filename, model_path=artifact_dir, overwrite=overwrite
                    )
                    labeled_frame = predictions[frame_idx]

                    # Custom visualization in the grid
                    ax = axes[file_idx, model_idx]
                    plot_custom_img(ax, labeled_frame.image)
                    plot_custom_instances(ax, labeled_frame.instances, lw=2, ms=50)

                    if model_idx == 0:
                        # Add filename label on the left side for each row
                        fig.text(
                            x=-0.01,  # Position outside the grid
                            y=(num_files - file_idx - 0.5)
                            / num_files,  # Center vertically in the row
                            s=Path(filename).stem,  # Filename without extension
                            ha="right",
                            va="center",
                            fontsize=(
                                font_settings.get("text_fontsize", 12)
                                if font_settings
                                else 12
                            ),
                            rotation=0,
                        )

                    # Add artifact name label at the bottom for each column
                    if file_idx == 0:
                        fig.text(
                            x=(model_idx + 0.5) / num_models,
                            y=-0.05,
                            s=artifact_name,
                            ha="right",
                            va="center",
                            fontsize=(
                                font_settings.get("text_fontsize", 12)
                                if font_settings
                                else 12
                            ),
                            rotation=45,
                        )

                except Exception as e:
                    logging.error(
                        f"Error processing artifact: {artifact_name}. Exception: {e}"
                    )

    # Apply custom spacing if provided
    spacing_defaults = {
        "left": 0.01,
        "right": 0.95,
        "top": 0.95,
        "bottom": 0.01,
        "wspace": 0.1,
        "hspace": 0.1,
    }
    spacing = spacing or spacing_defaults
    plt.subplots_adjust(**spacing)

    # Save the figure with customizable settings
    plt.savefig(
        Path(output_dir) / f"grid_predictions_frame_{frame_idx}.png",
        bbox_inches="tight",
        dpi=dpi,
        transparent=transparent,
    )
    print(
        f"Saved grid visualization to {output_dir}/grid_predictions_frame_{frame_idx}.png"
    )

    plt.show()
    plt.close()
    run.finish()


def predictions_viz_from_sleap_files(
    prediction_files_grid: List[List[Path]],
    test_group_names: List[str],
    model_names: List[str],
    output_path: Path,
    frame_idx: int = 1,
    figsize_scale: float = 5,
    dpi: int = 300,
    transparent: bool = False,
    font_settings: dict = None,
    colormap: str = "magma",
    spacing: dict = None,
):
    """
    Visualize predictions from multiple models on multiple test groups in a grid.

    Each row shows predictions from one model across all test groups.
    Each column is a test group. Cell (i, j) = model[i] on group[j].

    Args:
        prediction_files_grid (List[List[Path]]): 2D list [num_models][num_groups] of prediction `.slp` files.
        test_group_names (List[str]): Names of test groups (1 per column).
        model_names (List[str]): Names of models (1 per row).
        output_path (Path): Path to save the output figure.
        frame_idx (int): Frame index to display from each file.
        figsize_scale (float): Size multiplier for layout.
        dpi (int): Output resolution.
        transparent (bool): Save with transparent background.
        font_settings (dict): Optional font settings.
        colormap (str): Matplotlib colormap.
        spacing (dict): Subplot spacing.
    """
    output_path = Path(output_path)

    num_models = len(prediction_files_grid)
    num_groups = len(test_group_names)

    fig, axes = plt.subplots(
        num_models,
        num_groups,
        figsize=(figsize_scale * num_groups, figsize_scale * num_models),
    )

    if transparent:
        fig.patch.set_alpha(0)

    if font_settings:
        plt.rcParams.update(font_settings)

    plt.set_cmap(colormap)

    # If grid is 1 row or 1 col, make axes indexable
    if num_models == 1:
        axes = [axes]
    if num_groups == 1:
        axes = [[ax] for ax in axes]

    for row_idx, (model_name, row_files) in enumerate(
        zip(model_names, prediction_files_grid)
    ):
        for col_idx, (group_name, slp_path) in enumerate(
            zip(test_group_names, row_files)
        ):
            ax = axes[row_idx][col_idx]
            try:
                labels = sleap.load_file(slp_path)
                if labels is None:
                    print(f"No labels in {slp_path}")
                    continue
                if frame_idx >= len(labels):
                    print(f"Frame {frame_idx} out of range in {slp_path.name}")
                    continue

                labeled_frame = labels[frame_idx]
                plot_custom_img(ax, labeled_frame.image)
                plot_custom_instances(ax, labeled_frame.instances, lw=2, ms=50)

                # Column label: group name (bottom row only)
                if row_idx == 0:
                    fig.text(
                        x=(col_idx + 0.5) / num_groups,
                        y=0.00,
                        s=group_name,
                        ha="right",
                        va="center",
                        fontsize=(
                            font_settings.get("text_fontsize", 12)
                            if font_settings
                            else 12
                        ),
                    )

                # Row label: model name (left col only)
                if col_idx == 0:
                    fig.text(
                        x=-0.01,
                        y=(num_models - row_idx - 0.5) / num_models,
                        s=model_name,
                        ha="right",
                        va="center",
                        fontsize=(
                            font_settings.get("text_fontsize", 12)
                            if font_settings
                            else 12
                        ),
                    )

            except Exception as e:
                logging.error(
                    f"Error loading/plotting {slp_path} (model: {model_name}, group: {group_name}): {e}"
                )

    # Apply subplot spacing
    spacing_defaults = {
        "left": 0.01,
        "right": 0.95,
        "top": 0.95,
        "bottom": 0.05,
        "wspace": 0.15,
        "hspace": 0.1,
    }
    spacing = spacing or spacing_defaults
    plt.subplots_adjust(**spacing)

    plt.savefig(output_path, dpi=dpi, bbox_inches="tight", transparent=transparent)
    print(f"Saved SLEAP grid visualization to {output_path}")
    plt.show()
    plt.close()


def visualize_predictions_from_artifacts(
    model_artifact_name: str,
    test_artifact_name: str,
    output_dir: str = "output",
    num_frames: int = 5,
    overwrite: bool = False,
):
    """
    Fetch a model and test data from W&B artifacts, generate predictions,
    and save visualizations as PNGs for the specified number of frames.
    The resulting images and predictions are added as W&B artifacts if newly created.

    Args:
        model_artifact_name (str): Name of the model artifact.
        test_artifact_name (str): Name of the test dataset artifact.
        output_dir (str): Directory to save outputs. Default is "output".
        num_frames (int): Number of labeled frames to visualize.
        overwrite (bool): Whether to overwrite existing predictions and images.
    """
    PROJECT_NAME = CONFIG["project_name"]
    ENTITY_NAME = CONFIG["entity_name"]
    EXPERIMENT_NAME = CONFIG["experiment_name"]
    REGISTRY = CONFIG["registry"]

    run = wandb.init(
        project=PROJECT_NAME,
        entity=ENTITY_NAME,
        name=f"viz_{model_artifact_name}_on_{test_artifact_name}",
        job_type="visualization",
        group=EXPERIMENT_NAME,
    )

    try:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        predictions_path = (
            output_path
            / f"{model_artifact_name}_on_{test_artifact_name}_test_predictions.slp"
        )

        if predictions_path.exists() and not overwrite:
            print(f"Loading cached predictions from {predictions_path}")
            labels_pr = sleap.load_file(predictions_path.as_posix())
        else:
            model_artifact = fetch_model_artifact(
                run=run,
                entity_name=ENTITY_NAME,
                registry=REGISTRY,
                artifact_name=model_artifact_name,
                alias="latest",
            )
            test_artifact = fetch_model_artifact(
                run=run,
                entity_name=ENTITY_NAME,
                registry=REGISTRY,
                artifact_name=test_artifact_name,
                alias="latest",
            )

            test_data = get_test_data(test_artifact)
            num_labeled_frames = len(test_data)
            if num_frames > num_labeled_frames:
                raise ValueError(
                    f"Requested {num_frames} frames, but test set has only {num_labeled_frames} labeled frames."
                )

            model_dir = model_artifact.download(skip_cache=True)
            predictor = sleap.load_model(model_dir, progress_reporting="none")
            model = predictor.bottomup_model
            config = predictor.bottomup_config

            labels_pr, _ = sleap.nn.evals.evaluate_model(
                cfg=config, labels_gt=test_data, model=model, save=False
            )

            labels_pr.save(predictions_path.as_posix())
            print(f"Saved predictions to {predictions_path}")

        for i, labeled_frame in enumerate(labels_pr[:num_frames]):
            video_filename = (
                Path(labeled_frame.video.backend.filename).stem
                if labeled_frame.video
                else "unknown_video"
            )
            frame_index = labeled_frame.frame_idx
            img_title = f"{video_filename} | Frame {frame_index}"
            img_filename = f"{video_filename}_frame_{frame_index}_idx_{i}.png"
            img_path = output_path / img_filename

            if img_path.exists() and not overwrite:
                print(f"Skipping existing image: {img_path}")
                continue

            fig, ax = plt.subplots(figsize=(8, 8))
            plot_custom_img(ax, labeled_frame.image)
            plot_custom_instances(ax, labeled_frame.instances, lw=2, ms=50)
            fig.suptitle(img_title, fontsize=14)
            plt.savefig(img_path.as_posix(), dpi=300)
            plt.close()
            print(f"Saved prediction image: {img_path}")

        # Log entire directory as a W&B artifact
        images_artifact = wandb.Artifact(
            name=f"{model_artifact_name}_on_{test_artifact_name}_viz_outputs",
            type="predictions_visualizations",
            metadata={
                "model_artifact_name": model_artifact_name,
                "test_artifact_name": test_artifact_name,
                "num_frames": num_frames,
                "includes_predictions": predictions_path.exists(),
            },
        )
        images_artifact.add_dir(output_path.as_posix())
        run.log_artifact(images_artifact)
        print(f"Logged outputs as artifact: {images_artifact.name}")

    except Exception as e:
        print(f"Error during prediction visualization: {e}")
    finally:
        run.finish()


def plot_custom_img(ax, img: np.ndarray):
    """Plot an image onto a specific Matplotlib axis."""
    ax.imshow(
        img,
        cmap="gray",  # Use grayscale color map
        origin="upper",  # Set origin to upper left
    )
    ax.axis("off")  # Hide axis ticks and labels


def plot_custom_instances(ax, instances, skeleton=None, lw=2, ms=10, cmap=None):
    """Plot predictions (instances) onto a specific Matplotlib axis."""
    if cmap is None:
        cmap = plt.cm.tab10.colors  # Default color map

    for i, instance in enumerate(instances):
        points = instance.points_array  # Array of keypoints (x, y)
        color = cmap[i % len(cmap)]

        # Plot keypoints
        ax.scatter(points[:, 0], points[:, 1], color=color, s=ms)

        # Plot skeleton edges
        if skeleton:
            for src_node, dst_node in skeleton.edges:
                src_idx = skeleton.node_to_index[src_node]
                dst_idx = skeleton.node_to_index[dst_node]
                src_pt = points[src_idx]
                dst_pt = points[dst_idx]
                line = ConnectionPatch(
                    xyA=src_pt,
                    xyB=dst_pt,
                    coordsA="data",
                    coordsB="data",
                    axesA=ax,
                    axesB=ax,
                    color=color,
                    lw=lw,
                )
                ax.add_artist(line)


def fetch_sweep_metrics(
    sweep_ids: List[str],
    target_metrics: List[str],
    entity_name: Optional[str] = None,
    project_name: Optional[str] = None,
    include_config: bool = True,
) -> pd.DataFrame:
    """Fetch metrics from all runs in one or more W&B sweeps.

    Args:
        sweep_ids (List[str]): List of sweep IDs (e.g., ['abc1234', 'def5678']).
        target_metrics (List[str]): List of metric names to extract.
        entity_name (Optional[str]): The W&B entity. Defaults to CONFIG.
        project_name (Optional[str]): The W&B project name. Defaults to CONFIG.
        include_config (bool): Whether to include run config (e.g., input_scale).

    Returns:
        pd.DataFrame: DataFrame with one row per run, including metrics and configs.
    """

    entity = entity_name or CONFIG["entity_name"]
    project = project_name or CONFIG["project_name"]

    api = wandb.Api()
    records = []

    for sweep_id in sweep_ids:
        sweep = api.sweep(f"{entity}/{project}/{sweep_id}")

        for run in sweep.runs:
            if run.state != "finished":
                continue  # Skip incomplete runs

            row = {
                "run_id": run.id,
                "name": run.name,
                "group": run.group,
                "sweep_id": sweep_id,
            }

            for metric in target_metrics:
                row[metric] = run.summary.get(metric)

            if include_config:
                for key, value in run.config.items():
                    row[f"config/{key}"] = value

            records.append(row)

    df = pd.DataFrame(records)
    return df


def get_sweep_ids_for_group_from_runs(
    group_name: str,
    entity_name: Optional[str] = None,
    project_name: Optional[str] = None,
    filters: Optional[Dict] = None,
    earliest_time: Optional[str] = None,  # ISO 8601 format: "YYYY-MM-DDTHH:MM:SSZ"
) -> List[str]:
    """Retrieve unique W&B sweep IDs for a given run group, with support for extra filters and start time.

    Args:
        group_name (str): The run group name.
        entity_name (Optional[str]): The W&B entity name.
        project_name (Optional[str]): The W&B project name.
        filters (Optional[Dict]): Additional filters to apply (e.g., {"config.model_type": "resnet"}).
        earliest_time (Optional[str]): ISO 8601 timestamp (e.g., "2025-06-26T00:00:00Z").

    Returns:
        List[str]: Unique sweep IDs associated with the group and filters.
    """
    entity = entity_name or CONFIG["entity_name"]
    project = project_name or CONFIG["project_name"]

    query_filters = {"group": group_name}
    if filters:
        query_filters.update(filters)
    if earliest_time:
        # https://docs.wandb.ai/ref/python/public-api/api/#runs
        query_filters["createdAt"] = {"$gte": earliest_time}

    api = wandb.Api()
    runs = api.runs(f"{entity}/{project}", filters=query_filters)

    logging.info(
        f"Found {len(runs)} runs in group '{group_name}' with filters: {query_filters}"
    )
    if not runs:
        logging.warning(
            f"No runs found in group '{group_name}' with filters: {query_filters}. Returning empty sweep IDs."
        )
        return []

    sweep_ids = sorted({run.sweep.id for run in runs if run.sweep is not None})
    if not sweep_ids:
        logging.warning(
            f"No sweep IDs found for group '{group_name}' with filters: {query_filters}. Returning empty list."
        )
    else:
        logging.info(f"Found {len(sweep_ids)} unique sweep IDs: {sweep_ids}")
    return sweep_ids


def evaluate_model(
    model_artifact_name: str,
    test_artifact_name: str,
    output_dir: str = "output",
    px_per_mm=17.0,
) -> tuple:
    """Evaluate a model artifact against a test dataset.

    Args:
        model_artifact_name (str): The name of the model artifact to evaluate.
        test_artifact_name (str): The name of the test dataset artifact. This will be in
            a model directory from a model artifact since associated test sets are saved
            in model artifacts.
        output_dir (str): The directory to save the evaluation results. Default is "output".
        px_per_mm (float): The number of pixels per millimeter for the dataset. Default is 17.0.

    Returns:
        sleap.Labels: The predicted labels.
        dict: The evaluation metrics (without changing units).
        metrics_summary: Dictionary containing the summary metrics (units in mm).
    """
    PROJECT_NAME = CONFIG["project_name"]
    ENTITY_NAME = CONFIG["entity_name"]
    EXPERIMENT_NAME = CONFIG["experiment_name"]
    REGISTRY = CONFIG["registry"]

    # Initialize W&B run
    run = wandb.init(
        project=PROJECT_NAME,
        entity=ENTITY_NAME,
        name=f"evaluate_{model_artifact_name}_on_{test_artifact_name}",
        job_type="evaluation",
        group=EXPERIMENT_NAME,
    )

    try:
        # Fetch the model artifact
        model_artifact = fetch_model_artifact(
            run=run,
            entity_name=ENTITY_NAME,
            registry=REGISTRY,
            artifact_name=model_artifact_name,
            alias="latest",
        )

        # Fetch the test dataset artifact
        test_artifact = fetch_model_artifact(
            run=run,
            entity_name=ENTITY_NAME,
            registry=REGISTRY,
            artifact_name=test_artifact_name,
            alias="latest",
        )

        # Get test data from the test dataset artifact
        test_data = get_test_data(test_artifact)
        # Save the test data to a local file
        test_data.save(Path(output_dir) / f"{test_artifact_name}_test_labels.slp")
        print(f"Test data saved to {output_dir}/{test_artifact_name}_test_labels.slp")

        # Load the model
        model_dir = model_artifact.download(
            skip_cache=True
        )  # Download entire model directory
        predictor = sleap.load_model(
            model_dir, progress_reporting="none"
        )  # Returns `Predictor` object
        print(f"Model loaded successfully from {model_dir}.")
        model = predictor.bottomup_model  # Get the underlying `tf.keras.Model`
        print(f"Bottom-up model loaded successfully: {model}.")
        config = predictor.bottomup_config

        # Evaluate model
        labels_pr, metrics = sleap.nn.evals.evaluate_model(
            cfg=config,
            labels_gt=test_data,
            model=model,
            save=False,  # Do not save the predictions in the models directory
        )

        # Save predictions
        labels_pr.save(
            Path(output_dir)
            / f"{model_artifact_name}_on_{test_artifact_name}_test_predictions.slp"
        )
        print(
            f"Predictions saved to {output_dir}/{model_artifact_name}_on_{test_artifact_name}_test_predictions.slp"
        )

        # Log the predictions file
        predictions_artifact = wandb.Artifact(
            name=f"{model_artifact_name}_on_{test_artifact_name}_test_predictions",
            type="predictions",
            metadata={
                "model_artifact_name": model_artifact_name,
                "test_artifact_name": test_artifact_name,
            },
        )
        predictions_artifact.add_file(
            Path(output_dir)
            / f"{model_artifact_name}_on_{test_artifact_name}_test_predictions.slp"
        )

        # Extract summary metrics
        metrics_summary = {
            "model_path": model_dir,
            "model_name": Path(model_dir).name,
            "test_path": test_artifact.download(skip_cache=True),
            "test_set_name": test_artifact_name,
            "dist_p50": metrics["dist.p50"] / px_per_mm,
            "dist_p90": metrics["dist.p90"] / px_per_mm,
            "dist_p95": metrics["dist.p95"] / px_per_mm,
            "dist_p99": metrics["dist.p99"] / px_per_mm,
            "dist_avg": metrics["dist.avg"] / px_per_mm,
            "dist_std": np.nanstd(metrics["dist.dists"].flatten()) / px_per_mm,
            "vis_prec": metrics["vis.precision"],
            "vis_recall": metrics["vis.recall"],
            "oks_map": metrics["oks_voc.mAP"],
            "oks_mar": metrics["oks_voc.mAR"],
        }

        metrics_summary_df = pd.DataFrame([metrics_summary])
        metrics_summary_df.to_csv(
            Path(output_dir)
            / f"{model_artifact_name}_on_{test_artifact_name}_metrics.csv",
            index=False,
        )
        print(
            f"Metrics saved to {output_dir}/{model_artifact_name}_on_{test_artifact_name}_metrics.csv"
        )

        # Log the metrics file
        metrics_artifact = wandb.Artifact(
            name=f"{model_artifact_name}_on_{test_artifact_name}_metrics",
            type="metrics",
            metadata={
                "model_artifact_name": model_artifact_name,
                "test_artifact_name": test_artifact_name,
            },
        )
        metrics_artifact.add_file(
            Path(output_dir)
            / f"{model_artifact_name}_on_{test_artifact_name}_metrics.csv"
        )

        # Log metrics to W&B
        for metric_name, metric_value in metrics_summary_df.iloc[0].items():
            run.summary[metric_name] = metric_value

        # Save detailed distance metrics
        dists = metrics["dist.dists"].flatten() / px_per_mm
        dists_df = pd.DataFrame({"distances_mm": dists})
        dists_df.to_csv(
            Path(output_dir)
            / f"{model_artifact_name}_on_{test_artifact_name}_distances.csv",
            index=False,
        )
        print(
            f"Distances saved to {output_dir}/{model_artifact_name}_on_{test_artifact_name}_distances.csv"
        )
        # Log the distances file
        distances_artifact = wandb.Artifact(
            name=f"{model_artifact_name}_on_{test_artifact_name}_distances",
            type="distances",
            metadata={
                "model_artifact_name": model_artifact_name,
                "test_artifact_name": test_artifact_name,
            },
        )
        distances_artifact.add_file(
            Path(output_dir)
            / f"{model_artifact_name}_on_{test_artifact_name}_distances.csv"
        )

        # Generate histogram for distances
        plt.figure(figsize=(10, 6))
        sns.histplot(dists, bins=30, kde=True, color="blue")
        plt.axvline(
            metrics_summary["dist_p50"],
            color="green",
            linestyle="--",
            label="50th Percentile",
        )
        plt.axvline(
            metrics_summary["dist_p90"],
            color="orange",
            linestyle="--",
            label="90th Percentile",
        )
        plt.axvline(
            metrics_summary["dist_avg"],
            color="red",
            linestyle="--",
            label="Average Distance",
        )
        plt.title("Distribution of Distances")
        plt.xlabel("Distance (mm)")
        plt.ylabel("Frequency")
        plt.legend()
        histogram_path = (
            Path(output_dir)
            / f"{model_artifact_name}_on_{test_artifact_name}_distance_histogram.png"
        )
        plt.savefig(histogram_path)
        plt.close()
        print(f"Distance histogram saved to {histogram_path}")
        # Log the histogram image
        histogram_artifact = wandb.Artifact(
            name=f"{model_artifact_name}_on_{test_artifact_name}_distance_histogram",
            type="histogram",
            metadata={
                "model_artifact_name": model_artifact_name,
                "test_artifact_name": test_artifact_name,
            },
        )
        histogram_artifact.add_file(histogram_path)

        # Log artifacts
        run.log_artifact(predictions_artifact, type="predictions")
        run.log_artifact(metrics_artifact, type="metrics")
        run.log_artifact(distances_artifact, type="metrics")
        run.log_artifact(histogram_artifact, type="metrics")

        return labels_pr, metrics, metrics_summary

    except Exception as e:
        print(f"Error during model evaluation: {e}. Skipping this model...")
        run.finish()

        # Return safe empty objects
        return sleap.Labels() if hasattr(sleap, "Labels") else {}, {}, {}

    finally:
        run.finish()


def main(
    groups: List[str],
    versions: List[str],
    tags: Optional[List[str]],
    metrics_artifact_name: str = "summary_metrics",
    csv_path: str = "metrics.csv",
):
    """Main function to fetch model artifacts and save metrics to CSV.

    Args:
        groups (List[str]): List of group names to fetch artifacts from.
        versions (List[str]): List of train/test split versions to fetch for each group.
        tags (Optional[List[str]]): List of tags to add to the W&B run.
        metrics_artifact_name (str): Name of the artifact to save metrics. Default is "summary_metrics".
        csv_path (str): Path to save the CSV file. Default is "metrics.csv".

    Returns:
        pd.DataFrame: DataFrame containing the metrics or None if no metrics are found.
    """
    PROJECT_NAME = CONFIG["project_name"]
    ENTITY_NAME = CONFIG["entity_name"]
    EXPERIMENT_NAME = CONFIG["experiment_name"]
    REGISTRY = CONFIG["registry"]

    # Initialize W&B run
    run = wandb.init(
        project=PROJECT_NAME,
        entity=ENTITY_NAME,
        name=EXPERIMENT_NAME,
        job_type="fetch_metrics",
        tags=tags,
        group=EXPERIMENT_NAME,
    )

    # Make dictionary of artifact metrics
    metrics_dict = {}

    # Iterate over all groups and versions
    for group in groups:
        for version in versions:
            artifact_name = create_artifact_name(group, version)
            artifact = fetch_model_artifact(
                run=run,
                entity_name=ENTITY_NAME,
                registry=REGISTRY,
                artifact_name=artifact_name,
                alias="latest",
            )

            if artifact:
                metadata = artifact.metadata
                metrics_dict[artifact_name] = {
                    "group": group,
                    "version": version,
                    "dist_avg": metadata.get("dist_avg"),
                    "dist_p50": metadata.get("dist_p50"),
                    "dist_p90": metadata.get("dist_p90"),
                    "dist_p95": metadata.get("dist_p95"),
                    "dist_p99": metadata.get("dist_p99"),
                }

    # Convert metrics dictionary to dataframe
    if metrics_dict:
        metrics_df = pd.DataFrame.from_dict(metrics_dict, orient="index")
        metrics_df.to_csv(csv_path)
        print(f"Metrics saved to {csv_path}")

        # Create the artifact
        artifact = wandb.Artifact(
            name=metrics_artifact_name, type="metrics", metadata=metrics_dict
        )

        # Add the file to the artifact
        artifact.add_file(csv_path)

        # Log the artifact to the W&B run
        run.log_artifact(artifact)
        run.finish()
        return metrics_df
    else:
        print("No metrics found.")
        run.finish()
        return None
