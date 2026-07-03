"""Read-only audit of the sleap-roots-labels registry for missing embedded images.

Run via:
    conda run -n sleap_v1.4.1 --no-capture-output python audit.py \
        --output audit_results.csv
    conda run -n sleap_v1.4.1 --no-capture-output python audit.py \
        --collections soybean_primary_6nodes_v004_labels --search-paths Z:/users/...

Downloads each collection's latest version and reports embedding status +
recoverability tier. Writes nothing to the registry.
"""
from __future__ import annotations

import argparse

from sleap_roots_training.datasets import audit_registry


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", default=None)
    parser.add_argument("--entity", default=None)
    parser.add_argument("--collections", nargs="*", default=None)
    parser.add_argument("--all-versions", action="store_true")
    parser.add_argument("--search-paths", nargs="*", default=None)
    parser.add_argument("--download-root", default=None)
    parser.add_argument("--output", default="audit_results.csv")
    args = parser.parse_args()

    df = audit_registry(
        registry=args.registry,
        entity=args.entity,
        collections=args.collections,
        all_versions=args.all_versions,
        download_root=args.download_root,
        search_paths=args.search_paths,
    )

    if len(df) == 0:
        print("No artifacts matched.")
        return

    print(f"\nAudited {len(df)} artifact version(s).\n")
    cols = ["collection", "version", "embedded", "recoverable_via", "size_mb"]
    print(df[cols].to_string(index=False))
    print("\nEmbedded counts:")
    print(df["embedded"].value_counts().to_string())
    print("\nRecoverability tiers (broken artifacts):")
    print(df.loc[~df["embedded"], "recoverable_via"].value_counts().to_string())

    df.to_csv(args.output, index=False)
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
