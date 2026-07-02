"""Repair sleap-roots-labels registry packages that lack embedded images.

Dry-run by default (plans only). Add --apply to write new versions.

Run via:
    # Dry-run a single collection
    conda run -n sleap_v1.4.1 --no-capture-output python repair.py \
        --collection soybean_primary_6nodes_v004_labels --search-paths Z:/users/...
    # Apply
    conda run -n sleap_v1.4.1 --no-capture-output python repair.py \
        --collection soybean_primary_6nodes_v004_labels --search-paths Z:/users/... --apply
    # All broken collections (audits first)
    conda run -n sleap_v1.4.1 --no-capture-output python repair.py --collection all
"""
from __future__ import annotations

import argparse

from sleap_roots_training.datasets import audit_registry, repair_artifact


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection", required=True, help="Collection name or 'all'.")
    parser.add_argument("--registry", default=None)
    parser.add_argument("--entity", default=None)
    parser.add_argument("--apply", action="store_true", help="Write new versions.")
    parser.add_argument("--search-paths", nargs="*", default=None)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--download-root", default=None)
    args = parser.parse_args()

    if args.collection == "all":
        df = audit_registry(
            registry=args.registry,
            entity=args.entity,
            download_root=args.download_root,
            search_paths=args.search_paths,
        )
        targets = list(df.loc[~df["embedded"], "collection"])
        print(f"Broken collections to repair: {targets}")
    else:
        targets = [args.collection]

    for name in targets:
        result = repair_artifact(
            name,
            registry=args.registry,
            entity=args.entity,
            dry_run=not args.apply,
            search_paths=args.search_paths,
            out_dir=args.out_dir,
            download_root=args.download_root,
        )
        print(result)


if __name__ == "__main__":
    main()
