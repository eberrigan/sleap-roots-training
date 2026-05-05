"""Per Appendix Z Fix 13 — delete the partial W&B model artifact + run created
by the failed smoke training pass before re-launching."""
from __future__ import annotations

import wandb

ENTITY = "eberrigan-salk-institute-for-biological-studies"
PROJECT = "sleap-roots"
ARTIFACT_NAME = "soybean-aug-retrain-2026-05-01-lateral_v000"

api = wandb.Api()
print(f"=== Inspecting model artifact {ARTIFACT_NAME} ===")
try:
    coll = api.artifact_collection(type_name="model", name=f"{ENTITY}/{PROJECT}/{ARTIFACT_NAME}")
    versions = list(coll.artifacts())
    print(f"  {len(versions)} version(s)")
    for a in versions:
        print(f"  deleting {a.name}  size={a.size}  aliases={a.aliases}")
        a.delete(delete_aliases=True)
except Exception as e:
    print(f"  inspect/delete err: {type(e).__name__}: {e}")

print()
print("=== Inspecting/deleting orphan smoke runs (training_v000) ===")
runs = api.runs(f"{ENTITY}/{PROJECT}",
                filters={"display_name": "soybean-aug-retrain-2026-05-01-lateral_training_v000"})
for r in runs:
    print(f"  deleting run {r.id} ({r.name}) state={r.state}")
    r.delete()

print("\nDone.")
