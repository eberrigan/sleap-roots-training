"""Verify the W&B aliases applied by _alias_winners.py are actually live."""
from __future__ import annotations
import wandb

ENTITY = "eberrigan-salk-institute-for-biological-studies"
ORG = f"{ENTITY}-org"
PROJECT = "sleap-roots"
REGISTRY = "sleap-roots-models"

EXPECTED = [
    ("primary", 1, "preview-n1"),  # brightness winner
    ("lateral", 1, "preview-n1"),
    ("primary", 0, "none-baseline"),
    ("lateral", 0, "none-baseline"),
]

api = wandb.Api()
print("=== Project-side artifacts ===")
for model_type, version, expected_alias in EXPECTED:
    name = f"soybean-aug-retrain-2026-05-01-{model_type}_v00{version}"
    fq = f"{ENTITY}/{PROJECT}/{name}:latest"
    try:
        a = api.artifact(fq)
        ok = expected_alias in a.aliases
        mark = "✓" if ok else "✗"
        print(f"  {mark} {name}: aliases={a.aliases}  expected '{expected_alias}'")
    except Exception as e:
        print(f"  ✗ {name}: err {type(e).__name__}: {e}")

print("\n=== Registry-side ===")
for model_type, version, expected_alias in EXPECTED:
    coll = f"soybean-aug-retrain-2026-05-01-{model_type}"
    src_qual = f"{ENTITY}/{PROJECT}/{coll}_v00{version}"
    reg_path = f"{ORG}/wandb-registry-{REGISTRY}/{coll}"
    try:
        c = api.artifact_collection(type_name="model", name=reg_path)
        match = None
        for a in c.artifacts():
            if getattr(a, "source_qualified_name", "").startswith(src_qual):
                match = a
                break
        if match is None:
            print(f"  ✗ {coll}/v00{version}: no registry artifact found")
            continue
        ok = expected_alias in match.aliases
        mark = "✓" if ok else "✗"
        print(f"  {mark} {coll}/{match.name}: aliases={match.aliases}  expected '{expected_alias}'")
    except Exception as e:
        print(f"  ✗ {coll}/v00{version}: err {type(e).__name__}: {e}")
