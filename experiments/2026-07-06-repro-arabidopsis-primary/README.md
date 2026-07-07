# Reproduce the TF Arabidopsis-primary baseline

Onboarding task (Tier 0): retrain the existing TensorFlow SLEAP model
`cyl_arabidopsis_7-11DAG_primary_6nodes` end-to-end, evaluate on the held-out
test split, and record a documented **TF reference baseline** (config + metrics +
W&B run link). The new PyTorch (`sleap-nn`) pipeline will later be graded against
a fresh baseline; these TF numbers are the reference.

Scaffolded on a Mac but authored to run on **Windows + WSL2 with an RTX 5080**.
All paths are derived relative to this folder (no drive letters / network shares),
so it runs the same anywhere. Data + models are written under `data/` and are
git-ignored.

## Prerequisites (one time, on the training PC / WSL2)

1. Install SLEAP 1.4.1 and this package (see repo root `README.md`):
   ```bash
   conda create -y -n sleap -c conda-forge -c nvidia -c sleap/label/dev -c sleap -c anaconda sleap=1.4.1
   conda activate sleap
   cd <repo-root>            # the sleap-roots-training checkout
   pip install -e .[dev]
   wandb login
   ```
2. Get on the branch:
   ```bash
   git fetch origin
   git checkout anirudh/repro-arabidopsis-primary
   ```

## Run order

Everything lives in one notebook, `repro_arabidopsis_primary.ipynb`. Launch Jupyter
from inside this folder with the `sleap` env active and run the cells top to bottom:

```bash
conda activate sleap
cd experiments/2026-07-06-repro-arabidopsis-primary
jupyter lab   # open repro_arabidopsis_primary.ipynb
```

Sections, each mirroring an existing helper notebook:

| Section | What it does | Mirrors |
|---------|--------------|---------|
| 0. GPU check | Confirms TensorFlow can use the 5080 (gate). | (addition) |
| 1. Fetch labels | Downloads the labels artifact from W&B (with images). | `make_dataset_registry.ipynb` |
| 2. Make splits | Seed-pinned 80/10/10 train/val/test split + CSV. | `make_train_test_splits_first.ipynb` |
| 3. Modify config | Writes `initial_config_modified_v000.json` next to the split. | `modify_init_configs_second.ipynb` |
| 4. Train | `SMOKE_TEST=True` for a 2-epoch check, then full run. | `sleap_train_with_wandb_third.ipynb` |
| 5. Evaluate | Prints test-set metrics (the baseline numbers). | evaluate notebooks |

Then fill in `TF_REFERENCE_BASELINE.md` with the metrics + the W&B run URL.

## The RTX 5080 risk (read before step 4)

The 5080 is a Blackwell GPU (sm_120). SLEAP 1.4.1 ships an older TensorFlow/CUDA
build that predates Blackwell and may not be able to use it. If `0_gpu_check.py`
reports zero GPUs or a kernel error, **stop and message Elizabeth** with the
printed TF version rather than reducing batch size or grinding on it.

## Decisions to confirm (marked in code)

- **Config fidelity** (`3_make_config.py`): the committed `configs/initial_config.base.json`
  is a proven primary-root bottom-up config (from the Medicago-primary experiment
  in `tests/data`), NOT the exact Arabidopsis run's config. The original
  `20250625_cyl_arabidopsis_primary_receptive_field` experiment swept `max_stride`.
  To reproduce a specific original model faithfully, download that model's
  `training_config.json` from W&B, save it as
  `configs/reference_training_config.json`, and rerun step 3 (it prefers the
  reference automatically). Confirm the target model/`max_stride` with Elizabeth.
- **Split fractions** (`_common.py`): defaulted to 80/10/10, seed 42. Match the
  original if Elizabeth used different fractions.
- **Pixel scale** (`4_train.py`): metrics are recorded in pixels. Get the correct
  px/mm for this cyl-Arabidopsis setup before converting localization error to mm.

## Files

- `repro_arabidopsis_primary.ipynb` — the full pipeline (run cells in order).
- `configs/initial_config.base.json` — committed base training config.
- `configs/reference_training_config.json` — optional; drop the original run's
  `training_config.json` here and Section 3 will prefer it.
- `TF_REFERENCE_BASELINE.md` — the deliverable write-up (fill in after Section 5).
