"""Train all condition×seed runs of one model type via sleap_roots_training.train.main.

Run via:
    mamba run -n sleap_v1.4.1 --no-capture-output python 4_train.py primary
    mamba run -n sleap_v1.4.1 --no-capture-output python 4_train.py lateral
    mamba run -n sleap_v1.4.1 --no-capture-output python 4_train.py lateral --smoke-test

Calls `sleap_roots_training.train.main(csv_path=...)` which iterates the CSV
versions, opens each per-condition config, runs `sleap-train <config>`, and
registers the trained model as a W&B artifact.

**px_per_mm fix**: monkey-patches `srt_train.evaluate_model_and_generate_visuals`
default to 17.0 (lab-standard cylinder pixel scale) instead of None (which would
ship pixel-space dist_* metrics). Sentinel + assert ensures patch took effect.

**Registry-link**: passes `link_to_registry=False` (the package's built-in path
uses deprecated model-registry/* namespace). Linking to modern wandb-registry
namespace happens in 5_promote_winners.py post-step.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import sleap_roots_training as srt
import sleap_roots_training.train as srt_train
from sleap_roots_training.train import main as train_main


PX_PER_MM = 17.0  # Lab-standard cylinder pixel-to-mm conversion (matches deployed-2022 metadata)
ENTITY_NAME = "eberrigan-salk-institute-for-biological-studies"
PROJECT_NAME = "sleap-roots"
EXP_ROOT = Path(r"Z:/users/eberrigan/SLEAP/SLEAP_Soy/2026-05-01_aug_retrain")


def _patch_px_per_mm():
    """Monkey-patch evaluate_model_and_generate_visuals so its default px_per_mm is 17.0
    instead of None. Run_single_training calls this with px_per_mm=None hardcoded
    (train.py:524, 575); we override the default at the function level."""
    if getattr(srt_train, "_PATCHED_PX_PER_MM", False):
        return
    _orig = srt_train.evaluate_model_and_generate_visuals

    def _patched(model_dir, px_per_mm=PX_PER_MM):
        actual = px_per_mm if px_per_mm not in (None, 0) else PX_PER_MM
        return _orig(model_dir, px_per_mm=actual)

    srt_train.evaluate_model_and_generate_visuals = _patched
    srt_train._PATCHED_PX_PER_MM = True
    print(f"[patch] evaluate_model_and_generate_visuals default px_per_mm={PX_PER_MM}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_type", choices=["primary", "lateral"])
    parser.add_argument("--smoke-test", action="store_true",
                        help="Run only version=0 (single run)")
    args = parser.parse_args()

    _patch_px_per_mm()
    assert getattr(srt_train, "_PATCHED_PX_PER_MM", False), \
        "px_per_mm patch did not take effect — refusing to launch (would ship pixel-space dist_*)"

    working_dir = EXP_ROOT / args.model_type
    csv_path = working_dir / "train_test_splits.csv"
    assert csv_path.exists(), f"Missing CSV: {csv_path} (run 3_generate_condition_configs.py first)"

    experiment_name = f"soybean-aug-retrain-2026-05-01-{args.model_type}"
    base_tags = ["soybean", args.model_type,
                 "6nodes" if args.model_type == "primary" else "4nodes",
                 "aug-retrain", "2026-05-01"]

    srt.config.reset_config()
    srt.config.update_config(
        entity_name=ENTITY_NAME,
        project_name=PROJECT_NAME,
        experiment_name=experiment_name,
        registry="sleap-roots-models",
        collection_name=experiment_name,
        job_type="train",
    )

    if args.smoke_test:
        import pandas as pd
        df = pd.read_csv(csv_path)
        smoke_df = df[df["version"] == 0]
        smoke_csv = working_dir / "train_test_splits_smoke.csv"
        smoke_df.to_csv(smoke_csv, index=False)
        csv_path = smoke_csv
        base_tags = base_tags + ["smoke-test"]

    train_main(
        csv_path=str(csv_path),
        tags=base_tags,
        model_tags=base_tags,
        sleap_train_command="sleap-train {}",
        use_existing_model=False,
        use_sweep=False,
        link_to_registry=False,  # legacy model-registry/* path; linking happens in 5_promote_winners.py
        registry_name=None,
    )


if __name__ == "__main__":
    main()
