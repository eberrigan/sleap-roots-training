"""Produce ONE seed-pinned train/val/test split per model type for the soybean aug retrain.

Run via:
    mamba run -n sleap_v1.4.1 --no-capture-output python 2_make_splits.py primary
    mamba run -n sleap_v1.4.1 --no-capture-output python 2_make_splits.py lateral

Same CSV schema as `make_train_test_splits_first.ipynb` so downstream scripts
that consume `train_test_splits.csv` work unchanged. Splits are seed-pinned
so all condition configs of a model type share the same train/val/test.
"""
from __future__ import annotations

# Disable GPU for splits — only need CPU and want to avoid VRAM hold-over for next step.
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd
import sleap

SEED = 42
FRACTION_TRAIN = 0.9
VAL_TEST_FRACTION = 0.5  # 50/50 split of the 10% non-train

LABELS = {
    "primary": r"Z:/users/eberrigan/SLEAP/20250102_generalizability_experiment/primary/soybean/labels_soybean_primary_6nodes.v004.pkg.slp",
    "lateral": r"Z:/users/eberrigan/SLEAP/SLEAP_Soy/lateral_root_4_nodes/labels_soy_lateral_4nodes.v007.pkg.slp",
}
OUTPUT_BASE = Path(r"Z:/users/eberrigan/SLEAP/SLEAP_Soy/2026-05-01_aug_retrain")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_type", choices=list(LABELS.keys()))
    args = parser.parse_args()

    random.seed(SEED)
    np.random.seed(SEED)
    try:
        import tensorflow as tf
        tf.random.set_seed(SEED)
    except ImportError:
        pass

    out_dir = OUTPUT_BASE / args.model_type / "splits"
    out_dir.mkdir(parents=True, exist_ok=True)
    split_dir = out_dir / "train_test_split.v000"
    split_dir.mkdir(exist_ok=True)

    base = sleap.load_file(LABELS[args.model_type])
    user_labels = base.with_user_labels_only() if hasattr(base, "with_user_labels_only") else base
    print(f"Loaded {len(user_labels)} user-labeled frames from {LABELS[args.model_type]}")

    # 90/5/5 split
    labels_train, _, labels_remainder, _ = sleap.nn.data.training.split_labels_train_val(
        user_labels, 1 - FRACTION_TRAIN
    )
    labels_val, _, labels_test, _ = sleap.nn.data.training.split_labels_train_val(
        labels_remainder, VAL_TEST_FRACTION
    )

    train_path = split_dir / "train.pkg.slp"
    val_path = split_dir / "val.pkg.slp"
    test_path = split_dir / "test.pkg.slp"
    labels_train.save(train_path, with_images=True)
    labels_val.save(val_path, with_images=True)
    labels_test.save(test_path, with_images=True)

    # CSV in the same schema as the notebook so downstream consumers work unchanged
    rows = [
        {"path": train_path.as_posix(), "version": 0, "labeled_frames": len(labels_train), "split_type": "train"},
        {"path": val_path.as_posix(), "version": 0, "labeled_frames": len(labels_val), "split_type": "val"},
        {"path": test_path.as_posix(), "version": 0, "labeled_frames": len(labels_test), "split_type": "test"},
    ]
    csv_path = OUTPUT_BASE / args.model_type / "train_test_splits_base.csv"  # base = single split; per-condition CSV in 3_generate
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"  train: {len(labels_train)}  val: {len(labels_val)}  test: {len(labels_test)}")
    print(f"wrote {csv_path}")


if __name__ == "__main__":
    main()
