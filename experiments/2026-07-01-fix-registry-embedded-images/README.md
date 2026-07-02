# Fix missing embedded images in `sleap-roots-labels`

Many registry packages were logged without embedded image data, so they cannot be
used for remote training (SLEAP can't find the referenced videos on the training
machine). These scripts audit and repair them, and a guardrail
(`require_embedded_images=True` in `make_dataset_artifact`) prevents recurrence.

See the design doc: `docs/superpowers/specs/2026-07-01-fix-registry-embedded-images-design.md`.

## Audit (read-only)

```bash
conda run -n sleap_v1.4.1 --no-capture-output python audit.py --output audit_results.csv
```

Downloads each collection's `latest` version and reports, per artifact:
`embedded`, `recoverable_via`, and size. Writes a CSV; changes nothing.

## Recoverability tiers

- **already_ok** — the artifact already has embedded images; nothing to do.
- **already_embedded** — the artifact is broken, but the file at its
  `metadata["data_path"]` on Z: is embedded → re-register that file (no re-embedding).
- **referenced_videos** — re-embeddable by loading the artifact and
  `labels.save(with_images=True)`; the referenced videos must be reachable (use
  `--search-paths` to point at the directories where they now live — matched by
  basename).
- **none** — no pixel source found; needs manual attention.

## Repair (dry-run by default)

```bash
# Plan one collection (no writes)
conda run -n sleap_v1.4.1 --no-capture-output python repair.py \
    --collection soybean_primary_6nodes_v004_labels --search-paths Z:/users/eberrigan/...

# Apply (writes a new version into the same collection, becoming :latest)
conda run -n sleap_v1.4.1 --no-capture-output python repair.py \
    --collection soybean_primary_6nodes_v004_labels --search-paths Z:/users/eberrigan/... --apply

# Repair every broken collection (audits first)
conda run -n sleap_v1.4.1 --no-capture-output python repair.py --collection all --apply
```

**Note:** For a `referenced_videos` (tier-2) collection, even a dry-run downloads the
artifact and writes a re-embedded copy to a local temp dir (or `--out-dir`) to prove the
repair is achievable — it still writes nothing to the W&B registry; only `--apply`
registers a new version. Repair assumes the target collection is broken: run `audit.py`
first, or use `--collection all`, which repairs only collections the audit reports as
not embedded.

After applying, re-run `audit.py` to confirm every collection reports `embedded == True`.
