"""Generate N condition configs (4 conditions × 1 seed = 8 total per model type for n=1 preview).

Per-condition mini-experiment dir under EXP_ROOT/<model_type>/<cond>-seed<N>/
with symlinked/copied splits + initial_config_modified_v00<idx>.json.

Outputs a `train_test_splits.csv` per model type with 4 versions × 3 split_types = 12 rows
(n=1 preview; bump SEEDS to [0,1,2] for n=3 final).

Run via:
    mamba run -n sleap_v1.4.1 --no-capture-output python 3_generate_condition_configs.py
"""
from __future__ import annotations

import copy
import json
import os
import shutil
import sys
import zipfile
from pathlib import Path

import pandas as pd

EXP_ROOT = Path(r"Z:/users/eberrigan/SLEAP/SLEAP_Soy/2026-05-01_aug_retrain")
HELPERS = Path(r"c:/vaults/sleap-roots/2026-05-04-javier-soybean-aug-retrain/helpers")
COND_TABLE = Path(r"c:/vaults/sleap-roots/2026-05-04-javier-soybean-aug-retrain/condition_table.csv")

# Aug ranges (per Appendix B of spec):
# - brightness: pixel offset [-133.5, +10.0] / 255 = [-0.5234, +0.0392]
# - contrast (gamma): actual [0.5, 2.0] * 100 = [50, 200] (albumentations divides by 100)
CONDITIONS = {
    "none":       {"contrast": False, "brightness": False},
    "brightness": {"contrast": False, "brightness": True,
                   "brightness_min_val": -0.5234, "brightness_max_val": 0.0392},
    "contrast":   {"contrast": True,  "brightness": False,
                   "contrast_min_gamma": 50, "contrast_max_gamma": 200},
    "both":       {"contrast": True,  "brightness": True,
                   "contrast_min_gamma": 50, "contrast_max_gamma": 200,
                   "brightness_min_val": -0.5234, "brightness_max_val": 0.0392},
}
SEEDS = [0]  # n=1 preview; bump to [0,1,2] for n=3 final

DEPLOYED_ZIPS = {
    "primary": Path(r"Z:/users/eberrigan/20260401_Javier_Martinez_Pacheco_TTC_SALK_Soybean/models_downloader_output/221003_111420.multi_instance.n=1389.zip"),
    "lateral": Path(r"Z:/users/eberrigan/20260401_Javier_Martinez_Pacheco_TTC_SALK_Soybean/models_downloader_output/lateral_root_221006_172103.multi_instance.n=482.zip"),
}


def extract_base_config(zip_path: Path, out_path: Path):
    """Extract training_config.json from deployed model zip; this becomes our config template."""
    with zipfile.ZipFile(zip_path) as z:
        candidates = [n for n in z.namelist() if n.endswith("training_config.json")]
        if not candidates:
            raise FileNotFoundError(f"no training_config.json in {zip_path}")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(z.read(candidates[0]))


def main():
    cond_rows = []
    for model_type in ("primary", "lateral"):
        # Snapshot deployed config as the template
        base_path = HELPERS / f"deployed_{model_type}_training_config.json"
        if not base_path.exists():
            print(f"  Extracting deployed {model_type} config to {base_path}")
            extract_base_config(DEPLOYED_ZIPS[model_type], base_path)
        base_cfg = json.loads(base_path.read_text())

        # Locate the shared splits (output of 2_make_splits.py)
        split_dir = EXP_ROOT / model_type / "splits" / "train_test_split.v000"
        train_pkg = split_dir / "train.pkg.slp"
        val_pkg = split_dir / "val.pkg.slp"
        test_pkg = split_dir / "test.pkg.slp"
        for p in (train_pkg, val_pkg, test_pkg):
            if not p.exists():
                sys.exit(f"split missing: {p} — run 2_make_splits.py {model_type} first")

        csv_rows = []
        version = 0
        for cond_name in ("none", "brightness", "contrast", "both"):
            for seed in SEEDS:
                run_dir = EXP_ROOT / model_type / f"{cond_name}-seed{seed}"
                run_dir.mkdir(parents=True, exist_ok=True)

                # Copy splits to run_dir (symlinks are flaky on Windows networked drives)
                for src, name in ((train_pkg, "train.pkg.slp"), (val_pkg, "val.pkg.slp"), (test_pkg, "test.pkg.slp")):
                    dst = run_dir / name
                    if not dst.exists() or dst.stat().st_size != src.stat().st_size:
                        shutil.copy2(src, dst)

                # Modify base config
                cfg = copy.deepcopy(base_cfg)
                cfg["data"]["labels"]["training_labels"] = (run_dir / "train.pkg.slp").as_posix()
                cfg["data"]["labels"]["validation_labels"] = (run_dir / "val.pkg.slp").as_posix()
                cfg["data"]["labels"]["test_labels"] = (run_dir / "test.pkg.slp").as_posix()
                cfg["data"]["labels"]["training_inds"] = None
                cfg["data"]["labels"]["validation_inds"] = None
                cfg["data"]["labels"]["test_inds"] = None
                cfg["outputs"]["runs_folder"] = (run_dir / "models").as_posix()
                cfg["outputs"]["run_name"] = f"soybean-aug-retrain-2026-05-01-{model_type}-{cond_name}-seed{seed}"

                # Aug overrides — start from disabled defaults then apply condition
                aug = cfg["optimization"]["augmentation_config"]
                aug["contrast"] = False  # explicit reset
                aug["brightness"] = False
                aug.update(CONDITIONS[cond_name])
                cfg["optimization"]["seed"] = seed

                config_path = run_dir / f"initial_config_modified_v00{version}.json"
                config_path.write_text(json.dumps(cfg, indent=2))

                # CSV rows (3 per version: train/val/test)
                csv_rows.extend([
                    {"path": (run_dir / "train.pkg.slp").as_posix(), "version": version, "labeled_frames": 0, "split_type": "train"},
                    {"path": (run_dir / "val.pkg.slp").as_posix(), "version": version, "labeled_frames": 0, "split_type": "val"},
                    {"path": (run_dir / "test.pkg.slp").as_posix(), "version": version, "labeled_frames": 0, "split_type": "test"},
                ])
                cond_rows.append({"model_type": model_type, "version": version, "condition": cond_name,
                                  "seed": seed, "run_name": cfg["outputs"]["run_name"],
                                  "config_path": config_path.as_posix()})
                version += 1

        csv_path = EXP_ROOT / model_type / "train_test_splits.csv"
        pd.DataFrame(csv_rows).to_csv(csv_path, index=False)
        print(f"wrote {csv_path} ({len(csv_rows)} rows = {version} versions × 3 split_types)")

    pd.DataFrame(cond_rows).to_csv(COND_TABLE, index=False)
    print(f"\nwrote {COND_TABLE} ({len(cond_rows)} total)")


if __name__ == "__main__":
    main()
