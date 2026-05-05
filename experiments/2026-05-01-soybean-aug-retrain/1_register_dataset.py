"""Register soybean primary + lateral labels to W&B sleap-roots-labels registry.

Run via:
    mamba run -n sleap_v1.4.1 --no-capture-output python 1_register_dataset.py primary
    mamba run -n sleap_v1.4.1 --no-capture-output python 1_register_dataset.py lateral

Wraps `sleap_roots_training.datasets.make_dataset_artifact` per the canonical
`make_dataset_registry.ipynb` pattern. Same W&B side effects (artifact +
metadata + registry link) but reproducible. Registers to MODERN entity-scoped
registry (wandb-registry-sleap-roots-labels/<collection>).
"""
from __future__ import annotations

import argparse

import sleap_roots_training as srt
from sleap_roots_training.datasets import make_dataset_artifact


ENTITY_NAME = "eberrigan-salk-institute-for-biological-studies"
PROJECT_NAME = "sleap-roots"
REGISTRY = "sleap-roots-labels"

CONFIGS = {
    "primary": dict(
        artifact_name="soybean_primary_6nodes_v004_labels",
        collection_name="soybean_primary_6nodes_v004_labels",
        experiment_name="soybean_primary_6nodes_v004_labels_2026-05-01",
        description=("Soybean primary root labels (6 nodes, 1389 frames, v004). "
                     "Source: 20250102_generalizability_experiment. First soybean entry "
                     "in sleap-roots-labels registry; registered for 2026-05-01 aug retrain."),
        tags=["soybean", "primary", "6nodes", "v004", "2026-05-01-aug-retrain"],
        dataset_path=r"Z:/users/eberrigan/SLEAP/20250102_generalizability_experiment/primary/soybean/labels_soybean_primary_6nodes.v004.pkg.slp",
    ),
    "lateral": dict(
        artifact_name="soybean_lateral_4nodes_v007_labels",
        collection_name="soybean_lateral_4nodes_v007_labels",
        experiment_name="soybean_lateral_4nodes_v007_labels_2026-05-01",
        description=("Soybean lateral root labels (4 nodes, 482 frames, v007). "
                     "Source: SLEAP_Soy/lateral_root_4_nodes/. First soybean lateral entry."),
        tags=["soybean", "lateral", "4nodes", "v007", "2026-05-01-aug-retrain"],
        dataset_path=r"Z:/users/eberrigan/SLEAP/SLEAP_Soy/lateral_root_4_nodes/labels_soy_lateral_4nodes.v007.pkg.slp",
    ),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_type", choices=list(CONFIGS.keys()))
    args = parser.parse_args()
    cfg = CONFIGS[args.model_type]

    srt.config.reset_config()
    srt.config.update_config(
        entity_name=ENTITY_NAME,
        project_name=PROJECT_NAME,
        registry=REGISTRY,
        job_type="build_dataset",
        experiment_name=cfg["experiment_name"],
        collection_name=cfg["collection_name"],
    )
    artifact = make_dataset_artifact(
        artifact_name=cfg["artifact_name"],
        dataset_path=cfg["dataset_path"],
        link_to_registry=True,
        description=cfg["description"],
        tags=cfg["tags"],
    )
    print(f"OK: registered {cfg['artifact_name']} (size={artifact.size if hasattr(artifact, 'size') else '?'})")


if __name__ == "__main__":
    main()
