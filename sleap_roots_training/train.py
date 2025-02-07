import pandas as pd
import json
import wandb
import subprocess
import os
import matplotlib.pyplot as plt
import seaborn as sns
import sleap
import numpy as np
import logging

from typing import List, Optional, Dict
from pathlib import Path
from datetime import datetime

from sleap_roots_training.config import CONFIG

# Set up logging
logging.basicConfig(level=logging.INFO)


def load_training_data(csv_path):
    """Loads training data from a CSV file.

    Args:
        csv_path (Path): Path to the CSV file containing training data.

    Returns:
        pandas.DataFrame: DataFrame containing the training data.
    """
    return pd.read_csv(csv_path)


def get_training_groups(df):
    """Groups training data by version.

    Args:
        df (pandas.DataFrame): DataFrame containing the training data.

    Returns:
        pandas.core.groupby.DataFrameGroupBy: Grouped DataFrame.
    """
    return df.groupby("version")


def log_to_wandb(project_name, entity_name, experiment_name, version, config, config_path, tags=None):
    """Initializes a W&B run and logs the initial training configuration.

    Args:
        project_name (str): Name of the W&B project--group of experiments.
        entity_name (str): Name of the W&B entity--organization or user.
        experiment_name (str): Name of the experiment group.
        version (str): Version of the training run.
        config (dict): Configuration dictionary loaded from the JSON file.
        config_path (Path): Path to the training configuration file.
        tags (list, optional): List of tags to be added to the W&B run.

    Returns:
        wandb.Run: W&B run object.
    """
    run = wandb.init(
        project=project_name,
        entity=entity_name,
        group=experiment_name,
        config=config,
        name=f"{experiment_name}_training_v00{version}", # Unique name for the run
        tags=tags,
        mode="online",  # default
    )
    # Log the version and path to the config
    wandb.config.update({"version": version, "config_path": config_path.as_posix()})
    return run


def execute_training(command):
    """Executes the training command using subprocess.

    Args:
        command (str): Training command to be executed.

    Returns:
        None
    """
    print(f"Executing: {command}")
    try:
        # Run the command in a subprocess
        result = subprocess.run(
            command, 
            shell=True,           # Run the command through the shell
            check=True,           # Raise an exception if the command fails
            stdout=subprocess.PIPE,  # Capture standard output
            stderr=subprocess.PIPE,  # Capture standard error
            text=True             # Decode output as text (not bytes)
        )
        # Print real-time output to monitor training progress
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        # Handle errors by printing the stderr
        print(f"Error executing training command: {e.stderr}")
        raise


def log_model_artifact(run, experiment_name, model_tags, model_dir, version):
    """Logs a trained model as a W&B artifact and updates the W&B run config with the training configuration.

    Args:
        run (wandb.Run): The W&B run object.
        experiment_name (str): Name of the experiment group.
        model_tags (list): List of tags to be added to the model artifact.
        model_dir (Path): Path to the directory containing the trained model.
        version (str): Version of the training run.

    Returns:
        None
    """
    # Path to the training config
    training_config_path = model_dir / "training_config.json"
    training_config = {}

    # Load the training configuration if it exists
    if training_config_path.exists():
        with open(training_config_path, "r") as f:
            training_config = json.load(f)

        # Update the W&B run configuration
        run.config.update(training_config)
        print("W&B run configuration updated with training configuration.")

    # Create artifact
    # https://docs.wandb.ai/ref/python/artifact/
    model_artifact = wandb.Artifact(
        name=f"{experiment_name}_v00{version}",  # Unique name for the artifact
        type="model",
        metadata={
            "experiment": experiment_name,
            "version": version,
            **training_config,  # Add training config metadata if available
        },
    )

    # Add tags to the artifact
    for tag in model_tags:
        # Add tags as metadata
        model_artifact.metadata[tag] = True

    # Add the entire model directory to the artifact
    model_artifact.add_dir(model_dir)

    # Log the artifact to the W&B run
    run.log_artifact(model_artifact, type="model", tags=model_tags)
    print(f"Model artifact '{model_artifact.name}' logged to W&B.")


def evaluate_model_and_generate_visuals(model_dir, px_per_mm=17.0):
    """Evaluates the model and generates visualizations for metrics.

    Args:
        model_dir (str or Path): Path to the directory containing the trained model.
        px_per_mm (float): Pixel scaling factor for converting distances to mm.

    Returns:
        tuple:
            metrics_summary_df (pd.DataFrame): DataFrame containing summary metrics.
            dists_df (pd.DataFrame): DataFrame containing detailed distances.
            visualizations (dict): Dictionary of visualization names and file paths.
    """
    model_dir = Path(model_dir)
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found at: {model_dir}")
    
    model_dir_str = model_dir.as_posix()
    print(f"Model path: {model_dir_str}")

    # Load the model
    metrics = sleap.load_metrics(model_dir_str, split="test")
    print(f"Metrics loaded from model directory: {model_dir_str}")

    # Extract summary metrics
    metrics_summary = {
        "model_path": model_dir_str,
        "model_name":model_dir.name,
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

    # Save detailed distance metrics
    dists = metrics["dist.dists"].flatten() / px_per_mm
    dists_df = pd.DataFrame({"distances_mm": dists})

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
    histogram_path = model_dir / "distance_histogram.png"
    plt.savefig(histogram_path)
    plt.close()

    visualizations = {"distance_histogram": histogram_path}

    return metrics_summary_df, dists_df, visualizations


def log_model_artifact_with_evals(run, experiment_name, model_tags, model_dir, version, eval_fn, eval_args):
    """Logs a trained model as a W&B artifact, updates the W&B run config with the training configuration,
    and logs evaluation metrics and visualizations.

    Args:
        run (wandb.Run): The W&B run object.
        experiment_name (str): Name of the experiment group.
        model_tags (list): List of tags to be added to the model artifact.
        model_dir (Path): Path to the directory containing the trained model.
        version (str): Version of the training run.
        eval_fn (callable): Function to evaluate the model.
        eval_args (dict): Arguments required for the evaluation function.

    Returns:
        None
    """
    # Path to the training config
    training_config_path = model_dir / "training_config.json"
    training_config = {}

    # Load the training configuration if it exists
    if training_config_path.exists():
        with open(training_config_path, "r") as f:
            training_config = json.load(f)

        # Update the W&B run configuration
        run.config.update(training_config)
        print("W&B run configuration updated with training configuration.")

    # Create artifact
    model_artifact = wandb.Artifact(
        name=f"{experiment_name}_v00{version}",  # Unique name for the artifact
        type="model",
        metadata={
            "experiment": experiment_name,
            "version": version,
            **training_config,  # Add training config metadata if available
        },
    )

    # Add tags to the artifact
    for tag in model_tags:
        model_artifact.metadata[tag] = True

    # Add the entire model directory to the artifact
    model_artifact.add_dir(model_dir)

    # Perform model evaluation
    metrics_summary_df, dists_df, visualizations = eval_fn(**eval_args)

    # Save evaluation metrics as artifacts
    metrics_summary_csv_path = model_dir / "metrics_summary.csv"
    metrics_summary_df.to_csv(metrics_summary_csv_path, index=False)
    model_artifact.add_file(metrics_summary_csv_path)

    dists_csv_path = model_dir / "detailed_distances.csv"
    dists_df.to_csv(dists_csv_path, index=False)
    model_artifact.add_file(dists_csv_path)

    # Log metrics to W&B
    for metric_name, metric_value in metrics_summary_df.iloc[0].items():
        run.summary[metric_name] = metric_value
        model_artifact.metadata[metric_name] = metric_value

    # Log visualizations
    for viz_name, viz_path in visualizations.items():
        model_artifact.add_file(viz_path)

    # Log the artifact to the W&B run
    run.log_artifact(model_artifact, type="model", tags=model_tags)
    print(f"Model artifact '{model_artifact.name}' logged to W&B with evaluations.")


def update_config_with_wandb(config:dict) -> dict:
    """
    Updates the configuration dictionary with values from wandb.config dynamically.

    Args:
        config (dict): The existing training configuration.

    Returns:
        dict: Updated configuration with W&B parameters.
    """
    if wandb.config:
        logging.info("Updating configuration with W&B sweep parameters.")
        logging.info("Initial configuration:\n%s", json.dumps(config, indent=4))
        logging.info("W&B sweep parameters:\n%s", json.dumps(dict(wandb.config), indent=4))

        for key, value in wandb.config.items():
            keys = key.split(".")  # Convert "data.preprocessing.input_scaling" to ["data", "preprocessing", "input_scaling"]

            # Traverse the dictionary and set the value
            temp = config
            for subkey in keys[:-1]:  # Navigate to the correct nested level
                temp = temp.setdefault(subkey, {})  # Ensure intermediate dictionaries exist
            temp[keys[-1]] = value  # Set the final parameter

            logging.info(f"Updated parameter: {key} -> {value}")

        logging.info("Final updated configuration:\n%s", json.dumps(config, indent=4))
    
    return config


def process_training(project_name, entity_name, experiment_name, version, group, use_existing_model, sleap_train_command, tags=None, model_tags=None):
    """Processes a training run for a specific version.

    Args:
        project_name (str): Name of the W&B project--group of experiments.
        entity_name (str): Name of the W&B entity--organization or user.
        experiment_name (str): Name of the experiment group.
        version (str): Version of the training run.
        group (pandas.DataFrame): Group of rows corresponding to the version.
        use_existing_model (bool): Whether to use an existing model for evaluation.
        sleap_train_command (str): Training command to be executed.
        tags (list, optional): List of tags to be added to the W&B run.
        model_tags (list, optional): List of tags to be added to the model artifact.

    Returns:
        None
    """
    dir_path = Path(group.iloc[0]["path"]).parent
    logging.info(f"Directory path for version {version}: {dir_path}")

    config_path = dir_path / f"initial_config_modified_v00{version}.json"

    if not config_path.exists():
        logging.error(f"Config file not found for version {version}: {config_path}")
        raise FileNotFoundError(f"Config file not found for version {version}: {config_path}")

    # Load the original configuration (unchanged)
    with open(config_path, "r") as f:
        original_config = json.load(f)

    # Start W&B run
    run = log_to_wandb(
        project_name=project_name,
        entity_name=entity_name,
        experiment_name=experiment_name,
        version=version,
        config=original_config, 
        config_path=config_path,
        tags=tags
    )

    # Create a copy to modify (avoid modifying the original)
    config = json.loads(json.dumps(original_config))  # Deep copy

    # Update configuration dynamically from W&B (without modifying the original)
    config = update_config_with_wandb(config)

    # Generate a unique timestamp for this modified config
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    modified_config_path = dir_path / f"modified_config_v00{version}_{timestamp}.json"

    # Save modified config
    with open(modified_config_path, "w") as f:
        json.dump(config, f, indent=4)

    logging.info(f"Saved updated configuration to: {modified_config_path}")

    # Update wandb.config with the path to the modified config
    wandb.config.update({"config_path": modified_config_path.as_posix()}, allow_val_change=True)

    logging.info(f"Training with updated config for version {version}.")

    try:
        # Execute training command with the modified config
        command = sleap_train_command.format(modified_config_path.as_posix())
        logging.info(f"Executing training command: {command}")

        execute_training(command)

        # Log model and evaluation results
        model_dir = dir_path / "models"
        if model_dir.exists():
            log_model_artifact_with_evals(run, experiment_name, model_tags, model_dir, version, evaluate_model_and_generate_visuals, {"model_dir": model_dir, "px_per_mm": 17.0})
        else:
            logging.error(f"Model directory does not exist: {model_dir}")
            raise FileNotFoundError(f"Model directory does not exist: {model_dir}")

    except Exception as e:
        logging.error(f"Training failed for version {version}: {str(e)}", exc_info=True)

    finally:
        run.finish()
        logging.info(f"W&B run for version {version} finished.")


def main(
    csv_path: str,
    tags: Optional[List[str]] = None,
    model_tags: Optional[List[str]] = None,
    sleap_train_command: str = "sleap-train {}",
    use_existing_model: bool = False,
    use_sweep: bool = False,
    sweep_config: Optional[Dict] = None
):
    """Main function to process all training runs.

    Args:
        csv_path (str): Path to the CSV file containing train-test splits paths.
        tags (List[str], optional): List of tags for W&B runs.
        model_tags (List[str], optional): List of tags for W&B model artifacts.
        sleap_train_command (str, optional): Command template for running SLEAP training.
        use_existing_model (bool, optional): Whether to use an existing model for evaluation.
        use_sweep (bool, optional): Whether to run a W&B sweep instead of a single run.
        sweep_config (Dict, optional): W&B sweep configuration dictionary.
    """

    try:
        # Set defaults if None
        tags = tags or []
        model_tags = model_tags or []

        # Load experiment details from global CONFIG
        PROJECT_NAME = CONFIG["project_name"]
        ENTITY_NAME = CONFIG["entity_name"]
        EXPERIMENT_NAME = CONFIG["experiment_name"]

        logging.info(f"Starting main function with CSV: {csv_path}")

        # Load training data
        df = load_training_data(csv_path)
        grouped = get_training_groups(df)

        def train():
            """Function that W&B Sweep Agent calls to run each training run."""
            try:
                for version, group in grouped:

                    process_training(
                        project_name=PROJECT_NAME,
                        entity_name=ENTITY_NAME,
                        experiment_name=EXPERIMENT_NAME,
                        version=version,
                        group=group,
                        use_existing_model=use_existing_model,
                        sleap_train_command=sleap_train_command,
                        tags=tags,
                        model_tags=model_tags
                    )
            except Exception as e:
                logging.error(f"Error during training execution: {str(e)}", exc_info=True)
                raise

        if use_sweep:
            if not sweep_config:
                logging.error("Sweep config must be provided when use_sweep=True.")
                raise ValueError("Sweep config must be provided when use_sweep=True.")

            logging.info("Creating W&B sweep...")
            sweep_id = wandb.sweep(sweep_config, project=PROJECT_NAME)
            logging.info(f"Sweep created with ID: {sweep_id}")

            # Dynamically determine `count` (total parameter combinations)
            param_combinations = 1
            for param in sweep_config["parameters"].values():
                param_combinations *= len(param["values"])  # Multiply all parameter choices

            logging.info(f"Running W&B sweep with {param_combinations} experiments...")
            wandb.agent(sweep_id, function=train, count=param_combinations)

        else:
            logging.info("Running single training...")
            train()

        logging.info("All versions processed successfully.")

    except Exception as e:
        logging.error(f"Fatal error in main function: {str(e)}", exc_info=True)



