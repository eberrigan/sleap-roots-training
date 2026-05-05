"""Inspect + clean up the bad W&B artifacts created by the config rebind bug.

Bug recap: the first runs of 1_register_dataset.py used a buggy reset_config()
that left the imported CONFIG ref stale in datasets.py. As a result:
  - primary v0: uploaded correctly to project, but registry-link tried wandb-registry-None/None (404'd, no link created)
  - lateral v0: uploaded correctly to project, but registry-link went to PRIMARY's collection
    (`wandb-registry-sleap-roots-labels/soybean_primary_6nodes_v004_labels`) — wrong collection.

This script:
  1. Inspects the current state of the project + registry artifacts.
  2. Optionally deletes the broken state so a re-registration can recreate clean v0.

Run with --dry-run to inspect only; --delete to actually delete.
"""
from __future__ import annotations

import argparse
import wandb

ENTITY = "eberrigan-salk-institute-for-biological-studies"
PROJECT = "sleap-roots"
REGISTRY = "sleap-roots-labels"
DATASET_ARTIFACTS = ("soybean_primary_6nodes_v004_labels",
                     "soybean_lateral_4nodes_v007_labels")


def inspect(api):
    print("=== Registry collections under wandb-registry-sleap-roots-labels ===")
    for coll_name in DATASET_ARTIFACTS:
        path = f"{ENTITY}-org/wandb-registry-{REGISTRY}/{coll_name}"
        try:
            coll = api.artifact_collection(type_name="dataset", name=path)
            versions = list(coll.artifacts())
            print(f"  {coll_name}: {len(versions)} version(s)")
            for a in versions:
                src = getattr(a, "source_qualified_name", "?")
                print(f"    {a.name}  source={src}  aliases={a.aliases}")
        except Exception as e:
            print(f"  {coll_name}: NOT FOUND or err: {type(e).__name__}: {e}")

    print("\n=== Project-level dataset artifacts ===")
    for short in DATASET_ARTIFACTS:
        try:
            coll = api.artifact_collection(type_name="dataset", name=f"{ENTITY}/{PROJECT}/{short}")
            versions = list(coll.artifacts())
            print(f"  {short}: {len(versions)} version(s)")
            for a in versions:
                print(f"    {a.name}  size={a.size}  aliases={a.aliases}")
        except Exception as e:
            print(f"  {short}: NOT FOUND or err: {type(e).__name__}: {e}")


def delete_bad(api):
    print("=== Deleting bad registry-collection artifacts ===")
    # The wrongly-linked lateral inside primary's collection
    bad_path = f"{ENTITY}-org/wandb-registry-{REGISTRY}/soybean_primary_6nodes_v004_labels"
    try:
        coll = api.artifact_collection(type_name="dataset", name=bad_path)
        for a in coll.artifacts():
            print(f"  deleting registry artifact {a.name} (source={getattr(a, 'source_qualified_name', '?')})")
            a.delete(delete_aliases=True)
    except Exception as e:
        print(f"  registry inspect/delete err: {type(e).__name__}: {e}")

    print("\n=== Deleting project dataset artifacts (will be recreated cleanly) ===")
    for short in DATASET_ARTIFACTS:
        try:
            coll = api.artifact_collection(type_name="dataset", name=f"{ENTITY}/{PROJECT}/{short}")
            for a in list(coll.artifacts()):
                print(f"  deleting {a.name}")
                a.delete(delete_aliases=True)
        except Exception as e:
            print(f"  {short} delete err: {type(e).__name__}: {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["inspect", "delete"])
    args = parser.parse_args()
    api = wandb.Api()
    inspect(api)
    if args.mode == "delete":
        print("\n")
        delete_bad(api)
        print("\n=== After delete ===")
        inspect(api)


if __name__ == "__main__":
    main()
