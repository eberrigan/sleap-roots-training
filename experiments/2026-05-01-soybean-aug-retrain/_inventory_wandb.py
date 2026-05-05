"""Read-only inventory of W&B artifacts to confirm only the today-uploaded
dataset artifacts were affected. Lists ALL artifact types in the project AND
all models in the models registry, so we can confirm nothing else was touched."""
from __future__ import annotations

import wandb

ENTITY = "eberrigan-salk-institute-for-biological-studies"
PROJECT = "sleap-roots"


def list_project_artifacts(api):
    print(f"=== Project artifacts in {ENTITY}/{PROJECT} ===")
    for art_type in ("dataset", "model"):
        try:
            collections = api.artifact_type(art_type, project=f"{ENTITY}/{PROJECT}").collections()
            print(f"\n  --- type={art_type} ---")
            for coll in collections:
                versions = list(coll.artifacts())
                print(f"  {coll.name}: {len(versions)} version(s)")
                for a in versions[:5]:
                    print(f"    {a.name}  size={a.size}  aliases={a.aliases}  created={a.created_at}")
                if len(versions) > 5:
                    print(f"    ... ({len(versions) - 5} more)")
        except Exception as e:
            print(f"  type={art_type}: err: {type(e).__name__}: {e}")


def list_registry(api, registry_name: str):
    print(f"\n=== Registry: wandb-registry-{registry_name} ===")
    try:
        # The org-level registry path
        registry_path = f"{ENTITY}-org/wandb-registry-{registry_name}"
        # Use the `Registry` mode if available
        for art_type in ("dataset", "model"):
            try:
                # Per-type listing inside this registry:
                proj = api.project(f"wandb-registry-{registry_name}", entity=f"{ENTITY}-org")
                print(f"  project found: {proj.name}")
                # Iterate collections within
                for coll in proj.artifacts_collections():
                    versions = list(coll.artifacts())
                    print(f"  collection={coll.name}: {len(versions)} version(s)")
                    for a in versions[:5]:
                        src = getattr(a, "source_qualified_name", "?")
                        print(f"    {a.name}  source={src}  aliases={a.aliases}")
                break  # only need to traverse once
            except Exception as e:
                print(f"  art_type={art_type} err: {type(e).__name__}: {e}")
                break
    except Exception as e:
        print(f"  registry top-level err: {type(e).__name__}: {e}")


def main():
    api = wandb.Api()
    list_project_artifacts(api)
    list_registry(api, "sleap-roots-labels")
    list_registry(api, "sleap-roots-models")


if __name__ == "__main__":
    main()
