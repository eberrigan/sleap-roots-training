"""Workaround for promote_model_in_registry bug — adds aliases via W&B API directly.

Per decision.json: brightness wins for both primary + lateral.
n_seeds=1 → use 'preview-n1' alias (NOT 'production').
none-baseline (v000) gets 'none-baseline' alias for posterity.

Targets the entity-scoped modern registry: wandb-registry-sleap-roots-models.
"""
from __future__ import annotations

import wandb

ENTITY = "eberrigan-salk-institute-for-biological-studies"
ORG = f"{ENTITY}-org"
PROJECT = "sleap-roots"
REGISTRY = "sleap-roots-models"

# From decision.json + condition_table.csv:
#   brightness winner -> v001 (both primary + lateral)
#   none baseline    -> v000 (both primary + lateral)
TO_ALIAS = [
    ("primary", 1, "preview-n1"),
    ("lateral", 1, "preview-n1"),
    ("primary", 0, "none-baseline"),
    ("lateral", 0, "none-baseline"),
]

api = wandb.Api()
for model_type, version, alias in TO_ALIAS:
    coll_name = f"soybean-aug-retrain-2026-05-01-{model_type}"
    artifact_name = f"{coll_name}_v00{version}"
    # Address the artifact via the project (not registry) — aliases live there
    fq = f"{ENTITY}/{PROJECT}/{artifact_name}:latest"
    try:
        art = api.artifact(fq)
        if alias not in art.aliases:
            art.aliases.append(alias)
            art.save()
            print(f"  added alias '{alias}' to {fq}")
        else:
            print(f"  alias '{alias}' already present on {fq}")
    except Exception as e:
        print(f"  FAILED {fq}: {type(e).__name__}: {e}")

# Also alias the registry-side artifacts (the linked entries inside the
# wandb-registry-sleap-roots-models/<collection>) for consumer-facing use.
print()
print("=== Registry-side aliasing ===")
for model_type, version, alias in TO_ALIAS:
    coll_name = f"soybean-aug-retrain-2026-05-01-{model_type}"
    # Find the registry artifact whose source is project/<artifact>:v00<version>
    src_qual = f"{ENTITY}/{PROJECT}/{coll_name}_v00{version}"
    reg_path = f"{ORG}/wandb-registry-{REGISTRY}/{coll_name}"
    try:
        coll = api.artifact_collection(type_name="model", name=reg_path)
        match = None
        for a in coll.artifacts():
            asrc = getattr(a, "source_qualified_name", None)
            if asrc and asrc.startswith(src_qual):
                match = a
                break
        if match is None:
            print(f"  no registry artifact found for source={src_qual}")
            continue
        if alias not in match.aliases:
            match.aliases.append(alias)
            match.save()
            print(f"  added alias '{alias}' to registry {reg_path}/{match.name}")
        else:
            print(f"  alias '{alias}' already on registry {reg_path}/{match.name}")
    except Exception as e:
        print(f"  FAILED registry alias for {coll_name} v{version}: {type(e).__name__}: {e}")

print("\nDone.")
