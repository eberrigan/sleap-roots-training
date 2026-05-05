"""Promote winning + none-baseline models in the W&B sleap-roots-models registry.

Run via:
    mamba run -n sleap_v1.4.1 --no-capture-output python 5_promote_winners.py

Two phases:
1. Link all 8 trained model artifacts (n=1 preview) to the modern entity-scoped
   registry: wandb-registry-sleap-roots-models/<collection>. Uses
   `models.fetch_model_artifact_and_link_to_registry` (canonical pattern).
2. Promote the chosen winner per model_type with alias `production`; promote
   `none`-baseline with alias `none-baseline` for posterity.

Reads `analysis/decision.json` for {primary_winner: <cond>, lateral_winner: <cond>}
or {h0: true} (skip promotions; only do the registry-link).
"""
from __future__ import annotations

import json
from pathlib import Path

import sleap_roots_training as srt
from sleap_roots_training.models import (
    fetch_model_artifact_and_link_to_registry,
    promote_model_in_registry,
)


ENTITY_NAME = "eberrigan-salk-institute-for-biological-studies"
PROJECT_NAME = "sleap-roots"
REGISTRY = "sleap-roots-models"
DECISION_JSON = Path(r"c:/vaults/sleap-roots/2026-05-04-javier-soybean-aug-retrain/analysis/decision.json")
COND_TABLE = Path(r"c:/vaults/sleap-roots/2026-05-04-javier-soybean-aug-retrain/condition_table.csv")


def link_all_to_registry():
    """Phase 1: link all trained artifacts to the modern entity-scoped registry."""
    import pandas as pd
    cond = pd.read_csv(COND_TABLE)
    for _, row in cond.iterrows():
        coll_name = f"soybean-aug-retrain-2026-05-01-{row['model_type']}"
        artifact_name = f"{coll_name}_v00{int(row['version'])}"
        try:
            fetch_model_artifact_and_link_to_registry(
                project_name=PROJECT_NAME, entity_name=ENTITY_NAME,
                artifact_name=artifact_name,
                registry_name=REGISTRY, collection_name=coll_name,
            )
            print(f"  linked {artifact_name} -> wandb-registry-{REGISTRY}/{coll_name}")
        except Exception as e:
            print(f"  WARN: failed to link {artifact_name}: {e}")


def promote_winners():
    """Phase 2: alias winner and none-baseline per model type.

    Stage selection is n_seeds-aware:
      - n_seeds >= 3 → 'production' alias (full replication)
      - n_seeds < 3  → 'preview-n{N}' alias (NOT production-grade)

    This prevents accidentally shipping an n=1 single-seed model to downstream
    consumers that auto-promote `production` aliased artifacts. (Critical-review
    reviewer 3 finding #7.)
    """
    if not DECISION_JSON.exists():
        print(f"  No {DECISION_JSON} — skip promotion phase. Run 5_promote_winners.py again after analysis.")
        return
    decision = json.loads(DECISION_JSON.read_text())
    if decision.get("h0"):
        print("H0 fired — no winners to promote.")
        return

    n_seeds = int(decision.get("n_seeds", 1))
    winner_stage = "production" if n_seeds >= 3 else f"preview-n{n_seeds}"
    if winner_stage != "production":
        print(f"  ⚠ n_seeds={n_seeds} < 3 — using stage={winner_stage!r} instead of 'production' "
              f"(do NOT promote single-seed results as production-grade).")

    import pandas as pd
    cond = pd.read_csv(COND_TABLE)
    for model_type, key in (("primary", "primary_winner"), ("lateral", "lateral_winner")):
        winner_cond = decision.get(key)
        if not winner_cond:
            print(f"  {model_type}: no winner declared in decision.json")
            continue
        # Find the version for this (model_type, condition, seed=0) — n=1 preview
        match = cond[(cond["model_type"] == model_type) & (cond["condition"] == winner_cond)]
        if len(match) == 0:
            print(f"  {model_type}: no condition_table row for cond={winner_cond}")
            continue
        version = int(match.iloc[0]["version"])
        artifact_name = f"soybean-aug-retrain-2026-05-01-{model_type}_v00{version}"
        try:
            promote_model_in_registry(
                project_name=PROJECT_NAME, entity_name=ENTITY_NAME,
                registry_name=REGISTRY, artifact_name=artifact_name, stage=winner_stage,
            )
            print(f"  PROMOTED {model_type} winner {winner_cond}: {artifact_name} -> {winner_stage}")
        except Exception as e:
            print(f"  WARN: failed to promote {artifact_name}: {e}")

    # Posterity: also alias none-baselines
    for model_type in ("primary", "lateral"):
        match = cond[(cond["model_type"] == model_type) & (cond["condition"] == "none")]
        if len(match) == 0: continue
        version = int(match.iloc[0]["version"])
        artifact_name = f"soybean-aug-retrain-2026-05-01-{model_type}_v00{version}"
        try:
            promote_model_in_registry(
                project_name=PROJECT_NAME, entity_name=ENTITY_NAME,
                registry_name=REGISTRY, artifact_name=artifact_name, stage="none-baseline",
            )
            print(f"  PROMOTED {model_type} none-baseline: {artifact_name} -> none-baseline")
        except Exception as e:
            print(f"  WARN: failed to promote {artifact_name} (none-baseline): {e}")


def main():
    srt.config.reset_config()
    srt.config.update_config(
        entity_name=ENTITY_NAME, project_name=PROJECT_NAME,
        registry=REGISTRY, job_type="promote_registry_artifact",
    )
    print("=== Phase 1: Link all trained artifacts to wandb-registry ===")
    link_all_to_registry()
    print("\n=== Phase 2: Promote winners + baselines ===")
    promote_winners()


if __name__ == "__main__":
    main()
