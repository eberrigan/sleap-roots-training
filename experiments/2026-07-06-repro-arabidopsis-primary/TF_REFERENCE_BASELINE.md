# TF Reference Baseline — cyl Arabidopsis primary (6 nodes)

Documented TensorFlow reference for the SLEAP roots pipeline. This is the number
the new PyTorch (`sleap-nn`) pipeline will be compared against. Exact numeric
parity is NOT expected; this is a reference, not a pass/fail bar.

_Fill in the `TODO` fields after running the pipeline (see `README.md`)._

## What was reproduced

- **Model / dataset:** cyl Arabidopsis, 7–11 DAG, primary root, 6 nodes
- **Labels artifact (W&B):** `cyl_arabidopsis_7-11DAG_primary_6nodes_labels` (registry `sleap-roots-labels`)
- **Original experiment referenced:** `20250625_cyl_arabidopsis_primary_receptive_field`
- **Reproduction experiment name:** `anirudh-repro-cyl-arabidopsis-primary-2026-07-06`
- **Reproduced by:** Anirudh — TODO date
- **Trained vs. loaded:** retrained from scratch on local RTX 5080 (WSL2)

## Config

- **Model type / head:** bottom-up `multi_instance` (confmaps + PAFs)  <!-- TODO confirm vs original -->
- **Backbone:** UNet — `max_stride` = TODO, `filters` = 24, `output_stride` = 2
- **Input scaling:** TODO (base default 1.0)
- **Batch size / epochs:** 4 / TODO
- **Augmentations enabled:** TODO
- **Split:** 80/10/10, seed 42  <!-- TODO confirm matches original -->
- **Full config used:** `data/splits/train_test_split.v000/initial_config_modified_v000.json`
  (also logged to the W&B run config)

## Metrics (held-out test split)

Units: pixels (px). Convert to mm only with the confirmed px/mm for this setup.

| Metric | Value |
|--------|-------|
| Localization error p50 (`dist.p50`) | TODO |
| Localization error p90 (`dist.p90`) | TODO |
| Localization error p95 (`dist.p95`) | TODO |
| Localization error mean (`dist.avg`) | TODO |
| OKS mAP (`oks_voc.mAP`) | TODO |
| OKS mAR (`oks_voc.mAR`) | TODO |
| Visibility precision (`vis.precision`) | TODO |
| Visibility recall (`vis.recall`) | TODO |
| PCK @ TODO px threshold | TODO (optional; compute from `dist.dists`) |

## Links

- **W&B training run:** TODO (URL)
- **W&B model artifact:** TODO (`anirudh-repro-cyl-arabidopsis-primary-2026-07-06_v000`)
- **Original run for comparison:** TODO (URL)

## Notes / how close to the original

- TODO: how the reproduced numbers compare to the original logged run, and any
  differences in config, split, or environment (e.g. GPU/TF version on the 5080).
