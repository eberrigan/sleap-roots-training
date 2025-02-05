import wandb
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sleap
import numpy as np

from matplotlib.patches import ConnectionPatch
from pathlib import Path
from typing import List
from wandb.sdk.wandb_run import Run
from wandb.sdk.artifacts.artifact import Artifact

from sleap_roots_training.config import CONFIG


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


def fetch_model_artifact(run: wandb.run, entity_name: str, registry: str, artifact_name: str, alias: str="latest") -> wandb.Artifact:
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
    full_artifact_name = f"{entity_name}-org/wandb-registry-{registry}/{artifact_name}:{alias}"
    print(f"Fetching artifact: {full_artifact_name}")
    artifact = run.use_artifact(f"{full_artifact_name}")
    return artifact


def get_eval_metadata(artifact: wandb.Artifact, metadata_key: str="dist_avg") -> float:
    """Return model artifact metric from metadata.
    
    Args:
        artifact (wandb.Artifact): The model artifact.
        metadata_key (str): The key of the metric to retrieve. Default is "dist_avg".
        
    Returns:
        float: The metric value.
    """
    metadata = artifact.metadata
    return metadata.get(metadata_key)


def get_predictions(filename: str, model_path:str, overwrite=False) -> sleap.Labels:
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


def predictions_viz(output_dir: str, filename: str, groups:List[str], frame_idx: int = 1, model_version: str = "002", overwrite=False):
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
        group=EXPERIMENT_NAME
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
                alias="latest"
            )
            artifact_dir = artifact.download(skip_cache=False)
            print(f"Downloaded artifact: {artifact_name} to {artifact_dir}")
        except Exception as e:
            wandb.alert(f"Error fetching artifact: {artifact_name}. Exception: {str(e)}")
            continue

        if artifact_dir:
            try:
                predictions = get_predictions(filename, model_path=artifact_dir, overwrite=overwrite)
                labeled_frame = predictions[frame_idx]

                # Visualize predictions
                sleap.nn.viz.plot_img(labeled_frame.image, scale=1.0)
                sleap.nn.viz.plot_instances(labeled_frame.instances, lw=2, ms=50)
                plt.title(f"Predictions for {artifact_name} at frame {frame_idx}")
                plt.savefig(Path(output_dir) / f"{Path(filename).stem}_{artifact_name}_frame_{frame_idx}.png")
                plt.close()  # Close the plot to avoid memory issues
            except Exception as e:
                wandb.alert(f"Error processing artifact: {artifact_name}. Exception: {str(e)}")

    run.finish()


def predictions_viz_multiple_files(
    output_dir: str, filenames: list, groups: List[str], frame_idx: int = 1, model_version: str = "002", overwrite=False
    ):
        """Visualize predictions for multiple files and models in a grid.

        Args:
            output_dir (str): Path to save outputs.
            filenames (list): List of video file paths for predictions.
            groups (List[str]): List of group names to fetch model artifacts from.
            frame_idx (int): Frame index for visualization. Default is 1.
            model_version (str): Model version to fetch from the registry. Default is "002".
            overwrite (bool): Whether to overwrite existing predictions. Default is False.

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
            tags=groups, # Add group names as tags
            group=EXPERIMENT_NAME
        )

        Path(output_dir).mkdir(parents=True, exist_ok=True)

        num_files = len(filenames)
        num_models = len(groups)
        fig, axes = plt.subplots(num_files, num_models, figsize=(5 * num_models, 5 * num_files))

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
                        alias="latest"
                    )
                    artifact_dir = artifact.download(skip_cache=False)
                    print(f"Downloaded artifact: {artifact_name} to {artifact_dir}")
                except Exception as e:
                    print(f"Error fetching artifact: {artifact_name}. Exception: {e}")
                    continue

                if artifact_dir:
                    try:
                        predictions = get_predictions(filename, model_path=artifact_dir, overwrite=overwrite)
                        labeled_frame = predictions[frame_idx]

                        # Custom visualization in the grid
                        ax = axes[file_idx, model_idx]
                        plot_custom_img(ax, labeled_frame.image)
                        plot_custom_instances(ax, labeled_frame.instances, lw=2, ms=50)

                        if model_idx == 0:
                            # Add filename label on the left side for each row
                            fig.text(
                                x=-0.01,  # Position outside the grid
                                y=(num_files - file_idx - 0.5) / num_files, # Center vertically in the row
                                s=Path(filename).stem,  # Filename without extension
                                ha="right",
                                va="center",
                                fontsize=12,
                                rotation=0,
                            )

                        # Add artifact name label at the bottom for each column
                        if file_idx == 0:  # Only add artifact name once per column
                            fig.text(
                                x=(model_idx + 0.5) / num_models,  # Center horizontally in the column
                                y=-0.05,  # Slightly below the grid
                                s=artifact_name,  # Artifact name
                                ha="right", # Center horizontally
                                va="center",# Center vertically
                                fontsize=12,  # Font size for visibility
                                rotation=45,  # Angled for readability
                                )

                    except Exception as e:
                        print(f"Error processing artifact: {artifact_name}. Exception: {e}")

        plt.subplots_adjust(left=0.00, # Adjust left margin
                            right=1.0, # Adjust right margin
                            top=1.00, # Adjust top margin
                            bottom=0.00, # Adjust bottom margin
                            wspace=0.00, # Adjust width space between subplots
                            hspace=0.00) # Adjust height space between subplots
        plt.savefig(Path(output_dir) / f"grid_predictions_frame_{frame_idx}.png", bbox_inches='tight', dpi=300) # Save the figure with tight bounding box
        print(f"Saved grid visualization to {output_dir}/grid_predictions_frame_{frame_idx}.png")
        plt.show()  # Show the plot
        plt.close()
        run.finish()


def plot_custom_img(ax, img: np.ndarray):
    """Plot an image onto a specific Matplotlib axis."""
    ax.imshow(
        img,
        cmap="gray", # Use grayscale color map
        origin="upper", # Set origin to upper left
    )
    ax.axis("off") # Hide axis ticks and labels


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
                    xyA=src_pt, xyB=dst_pt, coordsA="data", coordsB="data",
                    axesA=ax, axesB=ax, color=color, lw=lw
                )
                ax.add_artist(line)


def evaluate_model(model_artifact_name: str, test_artifact_name: str, output_dir: str="output", px_per_mm=17.0) -> tuple:
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
        dict: The evaluation metrics.
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
        group=EXPERIMENT_NAME
    )

    try:
        # Fetch the model artifact
        model_artifact = fetch_model_artifact(
            run=run,
            entity_name=ENTITY_NAME,
            registry=REGISTRY,
            artifact_name=model_artifact_name,
            alias="latest"
        )

        # Fetch the test dataset artifact
        test_artifact = fetch_model_artifact(
            run=run,
            entity_name=ENTITY_NAME,
            registry=REGISTRY,
            artifact_name=test_artifact_name,
            alias="latest"
        )

        # Get test data from the test dataset artifact
        test_data = get_test_data(test_artifact)
        # Save the test data to a local file
        test_data.save(Path(output_dir) / f"{test_artifact_name}_test_labels.slp")
        print(f"Test data saved to {output_dir}/{test_artifact_name}_test_labels.slp")

        # Load the model
        model_dir = model_artifact.download(skip_cache=True)  # Download entire model directory
        predictor = sleap.load_model(model_dir, progress_reporting="none") # Returns `Predictor` object
        print(f"Model loaded successfully from {model_dir}.")
        model = predictor.bottomup_model # Get the underlying `tf.keras.Model`
        print(f"Bottom-up model loaded successfully: {model}.")
        config = predictor.bottomup_config

        # Evaluate model
        labels_pr, metrics = sleap.nn.evals.evaluate_model(
            cfg=config, 
            labels_gt=test_data,
            model=model,
            save=False, # Do not save the predictions in the models directory

            )

        # Save predictions
        labels_pr.save(Path(output_dir) / f"{model_artifact_name}_on_{test_artifact_name}_test_predictions.slp")
        print(f"Predictions saved to {output_dir}/{model_artifact_name}_on_{test_artifact_name}_test_predictions.slp")

        # Log the predictions file
        predictions_artifact = wandb.Artifact(
            name=f"{model_artifact_name}_on_{test_artifact_name}_test_predictions",
            type="predictions",
            metadata={
                "model_artifact_name": model_artifact_name,
                "test_artifact_name": test_artifact_name
            }
        )
        predictions_artifact.add_file(Path(output_dir) / f"{model_artifact_name}_on_{test_artifact_name}_test_predictions.slp")

        # Extract summary metrics
        metrics_summary = {
            "model_path": model_dir,
            "model_name":Path(model_dir).name,
            "dist_p50": metrics["dist.p50"] / px_per_mm,
            "dist_p90": metrics["dist.p90"] / px_per_mm,
            "dist_p95": metrics["dist.p95"] / px_per_mm,
            "dist_p99": metrics["dist.p99"] / px_per_mm,
            "dist_avg": metrics["dist.avg"] / px_per_mm,
            "dist_std": np.nanstd(metrics["dist.dists"].flatten()) / px_per_mm,
            "vis_prec": metrics["vis.precision"],
            "vis_recall": metrics["vis.recall"],
            "oks_map": metrics["oks_voc.mAP"],
            "oks_mar": metrics["oks_voc.mAR"]
        }

        metrics_summary_df = pd.DataFrame([metrics_summary])
        metrics_summary_df.to_csv(Path(output_dir) / f"{model_artifact_name}_on_{test_artifact_name}_metrics.csv", index=False)
        print(f"Metrics saved to {output_dir}/{model_artifact_name}_on_{test_artifact_name}_metrics.csv")
        
        # Log the metrics file
        metrics_artifact = wandb.Artifact(
            name=f"{model_artifact_name}_on_{test_artifact_name}_metrics",
            type="metrics",
            metadata={
                "model_artifact_name": model_artifact_name,
                "test_artifact_name": test_artifact_name
            }
        )
        metrics_artifact.add_file(Path(output_dir) / f"{model_artifact_name}_on_{test_artifact_name}_metrics.csv")

        # Log metrics to W&B
        for metric_name, metric_value in metrics_summary_df.iloc[0].items():
            run.summary[metric_name] = metric_value
        
        # Save detailed distance metrics
        dists = metrics["dist.dists"].flatten() / px_per_mm
        dists_df = pd.DataFrame({"distances_mm": dists})
        dists_df.to_csv(Path(output_dir) / f"{model_artifact_name}_on_{test_artifact_name}_distances.csv", index=False)
        print(f"Distances saved to {output_dir}/{model_artifact_name}_on_{test_artifact_name}_distances.csv")
        # Log the distances file
        distances_artifact = wandb.Artifact(
            name=f"{model_artifact_name}_on_{test_artifact_name}_distances",
            type="distances",
            metadata={
                "model_artifact_name": model_artifact_name,
                "test_artifact_name": test_artifact_name
            }
        )
        distances_artifact.add_file(Path(output_dir) / f"{model_artifact_name}_on_{test_artifact_name}_distances.csv")

        # Generate histogram for distances
        plt.figure(figsize=(10, 6))
        sns.histplot(dists, bins=30, kde=True, color="blue")
        plt.axvline(metrics_summary["dist_p50"], color="green", linestyle="--", label="50th Percentile")
        plt.axvline(metrics_summary["dist_p90"], color="orange", linestyle="--", label="90th Percentile")
        plt.axvline(metrics_summary["dist_avg"], color="red", linestyle="--", label="Average Distance")
        plt.title("Distribution of Distances")
        plt.xlabel("Distance (mm)")
        plt.ylabel("Frequency")
        plt.legend()
        histogram_path = Path(output_dir) / f"{model_artifact_name}_on_{test_artifact_name}_distance_histogram.png"
        plt.savefig(histogram_path)
        plt.close()
        print(f"Distance histogram saved to {histogram_path}")
        # Log the histogram image
        histogram_artifact = wandb.Artifact(
            name=f"{model_artifact_name}_on_{test_artifact_name}_distance_histogram",
            type="histogram",
            metadata={
                "model_artifact_name": model_artifact_name,
                "test_artifact_name": test_artifact_name
            }
        )
        histogram_artifact.add_file(histogram_path)

        # Log artifacts
        run.log_artifact(predictions_artifact, type="predictions")
        run.log_artifact(metrics_artifact, type="metrics")
        run.log_artifact(distances_artifact, type="metrics")
        run.log_artifact(histogram_artifact, type="metrics")

        return labels_pr, metrics

    except Exception as e:
        print(f"Error during model evaluation: {e}. Skipping this model...")
        run.finish()
        
        # Return safe empty objects
        return sleap.Labels() if hasattr(sleap, "Labels") else {}, {}

    finally:
        run.finish()


def main(groups: List[str], versions: List[str], csv_path: str="metrics.csv"):
    """Main function to fetch model artifacts and save metrics to CSV.

    Args:
        groups (List[str]): List of group names to fetch artifacts from.
        versions (List[str]): List of train/test split versions to fetch for each group.
        csv_path (str): Path to save the CSV file. Default is "metrics.csv".

    Returns:
        pd.DataFrame: DataFrame containing the metrics or None if no metrics are found.
    """
    PROJECT_NAME = CONFIG["project_name"]
    ENTITY_NAME = CONFIG["entity_name"]
    EXPERIMENT_NAME = CONFIG["experiment_name"]
    REGISTRY = CONFIG["registry"]

    # Initialize W&B run
    run = wandb.init(project=PROJECT_NAME, 
                     entity=ENTITY_NAME, 
                     name=EXPERIMENT_NAME, 
                     job_type="fetch_metrics", 
                     tags=groups, 
                     group=EXPERIMENT_NAME
                     )

    # Make dictionary of artifact metrics
    metrics_dict = {}

    # Iterate over all groups and versions
    for group in groups:
        for version in versions:
            artifact_name = create_artifact_name(group, version)
            artifact = fetch_model_artifact(run=run, entity_name=ENTITY_NAME, registry=REGISTRY, artifact_name=artifact_name, alias="latest")
            
            if artifact:
                metadata = artifact.metadata
                metrics_dict[artifact_name] = {
                    "group": group,
                    "version": version,
                    "dist_avg": metadata.get("dist_avg"),
                    "dist_p50": metadata.get("dist_p50"),
                    "dist_p90": metadata.get("dist_p90"),
                    "dist_p95": metadata.get("dist_p95"),
                    "dist_p99": metadata.get("dist_p99")
                }

    # Convert metrics dictionary to dataframe
    if metrics_dict:
        metrics_df = pd.DataFrame.from_dict(metrics_dict, orient="index")
        metrics_df.to_csv(csv_path)
        print(f"Metrics saved to {csv_path}")

        # Create the artifact
        artifact = wandb.Artifact(
            name="summary_metrics",
            type="metrics",
            metadata=metrics_dict
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