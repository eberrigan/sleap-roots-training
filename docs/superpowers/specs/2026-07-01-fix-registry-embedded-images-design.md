# Design: Fix missing embedded images in the `sleap-roots-labels` registry

**Date:** 2026-07-01
**Status:** Approved (design)
**Author:** Elizabeth (with Claude Code)

## Problem

Many label packages in the W&B `sleap-roots-labels` registry lack **embedded image
data**. A SLEAP `.slp` without embedded pixels only *references* external video/image
files; when such an artifact is downloaded onto a training machine those files are
absent, so SLEAP cannot read frames and **training is impossible**.

The current registration path,
[`make_dataset_artifact`](../../../sleap_roots_training/datasets.py), does
`artifact.add_file(dataset_path)` with **no check** that the file carries embedded
images. Any file — a proper `.pkg.slp` saved `with_images=True`, or a stripped `.slp`
that only references external videos — is logged as-is. That is how broken packages
entered the registry.

## Goals

1. **Repair** existing registry artifacts that lack embedded images, re-registering
   trainable versions.
2. **Guardrail**: prevent non-embedded packages from being registered in future.

Both were explicitly requested.

## Non-goals

- Auditing/repairing the **models** registry (labels only).
- A generalized multi-registry "doctor" tool (YAGNI).
- Inventing pixels where no source exists (unrecoverable artifacts are reported, not
  fabricated).

## Key domain facts (empirically grounded)

Grounded against a real embedded package on Z: and the installed `sleap_v1.4.1` /
`wandb 0.18.7`:

- **Detection signal.** In an embedded package each `Video.backend` is an `HDF5Video`
  whose `.filename` is the package itself, with **`.has_embedded_images == True`** and
  `.embedded_frame_inds`. Embedded pixels live in per-video `videoN/video` HDF5
  datasets. A non-embedded `.slp` instead has `MediaVideo`/`ImageVideo` backends
  pointing at external files (no `has_embedded_images`).
- **Dead `source_video` ≠ broken.** An embedded package also records a `source_video`
  provenance path. That path can be long dead (e.g. a real soybean package references a
  gone `C:/Users/pbiobgh/Box/.../1WFWZA8J.h5`) yet `has_embedded_images` is still
  `True` and the package trains fine, because pixels are read from the embedding, not
  the source. Detection therefore checks `has_embedded_images`, **not** whether
  referenced files exist — a naive file-existence audit would wrongly flag good
  packages.
- **Save API (verified in `sleap/io/dataset.py`).**
  `Labels.save(filename, with_images=False, embed_all_labeled=False, embed_suggested=False)`.
  `with_images=True` embeds image data for frames that have (user) labels — exactly the
  training set. This matches the repo's existing usage in
  [`2_make_splits.py`](../../../experiments/2026-05-01-soybean-aug-retrain/2_make_splits.py).
  Repairs use plain `with_images=True` (not `embed_all_labeled`/`embed_suggested`) to
  stay content-equivalent to how training splits are built today. `.pkg.slp` filenames
  pass through `save()` unchanged (only a trailing `.slp` is required).
- **Native filepath remapping (verified in `sleap/io/dataset.py`).**
  `sleap.load_file(filename, detect_videos=True, search_paths=None, match_to=None)`
  relocates missing videos: `search_paths` is a path or list of paths (a video file or
  its containing folder), and the underlying `find_path_using_paths(missing_path,
  search_paths)` matches **by basename** (handling mixed `/` and `\`), returning the
  first hit. `Labels.load_file(..., video_search=<Callable | List[str]>)` also accepts a
  **callable** for arbitrary matching. This is the same mechanism the GUI's "locate
  missing videos" flow uses and is preferred over hand-rolled prefix remapping — for our
  case (a `.slp` referencing a gone `C:/Users/pbiobgh/Box/.../1WFWZA8J.h5` whose real
  file lives on Z:), pointing `search_paths` at the Z: directory finds the video by
  basename without needing to know the exact stale prefix.
- **Registry enumeration (wandb 0.18.7).** `api.registries()` does **not** exist in
  0.18.7 (that is 0.19+). The registry is a project
  `wandb-registry-sleap-roots-labels` under org entity `<entity>-org`. Working calls:
  - `api.artifact_collections("<entity>-org/wandb-registry-sleap-roots-labels", "dataset")`
    → collections
  - `collection.artifacts()` → versions; each version has `metadata["data_path"]`
    (Z: source path) and `aliases` (e.g. `latest`).
- **Current registry contents (8 dataset collections):**
  `soybean_lateral_4nodes_v007_labels`, `soybean_primary_6nodes_v004_labels`,
  `plate_medicago_14DAG_primary_8nodes_labels`,
  `plate_arabidopsis_2-7DAG_primary_8nodes_labels`,
  `cyl_arabidopsis_7-11DAG_primary_6nodes_labels`, `rice_3DAG_crown_6nodes_labels`,
  `wheat_5-14DAG_seminal_6nodes_labels`, `sorghum_5-12DAG_primary_6nodes_labels`.

## Approach (chosen: A)

Reusable, tested library functions in `sleap_roots_training/datasets.py` (preserving the
one-test-file-per-module convention → `tests/test_datasets.py`) plus a guardrail on the
existing registration path, driven by a thin CLI under `experiments/`. Rejected: a
one-off script (no guardrail, no reuse, no tests) and a generalized multi-registry
module (scope creep).

## Components — all in `sleap_roots_training/datasets.py`

`sleap` is imported **lazily inside** the functions that need it (not at module top),
so the module stays importable without SLEAP and the cross-platform `test-imports.yml`
job keeps working.

| Function | Purpose |
|---|---|
| `has_embedded_images(path) -> bool` | Single source of truth for "trainable on its own." For every video carrying user-labeled frames, requires `HDF5Video` + `has_embedded_images == True`. Returns `False` if any labeled-frame video lacks embedded pixels, or if there are **no** user-labeled frames. |
| `inspect_package(path) -> dict` | Richer per-file report: `embedded`, `n_user_frames`, per-video `{backend_type, embedded, referenced_path, referenced_exists}`, and `recoverable_via`. |
| `audit_registry(...) -> pd.DataFrame` | Enumerate collections, inspect each collection's `latest` (default; all-versions optional), build the report table below. |
| `repair_artifact(collection, *, dry_run=True, search_paths=None, video_search=None)` | Re-embed + re-register a fixed version into the same collection. |
| `make_dataset_artifact(..., require_embedded_images=True)` | Guardrail (see below). |

### Detection semantics (`has_embedded_images`)

- "Trainable" ≙ every user-labeled frame's pixels are available *from the package
  alone*. Implemented per-video: any video with user-labeled frames must be an
  `HDF5Video` with `has_embedded_images == True`.
- A dead `source_video` reference does not count as broken.
- Zero user-labeled frames → `False` (nothing to train on) and flagged by the audit.

### Recoverability ranking (per broken artifact, best → worst)

1. **`already_embedded`** — the file at the artifact's `metadata["data_path"]` on Z:
   exists and `has_embedded_images(that_file)` is `True`. The wrong (stripped) file was
   uploaded; the good one still sits on disk. Fix = re-register that file, **no image
   processing**. Cheapest and safest.
2. **`referenced_videos`** — no embedded pixels, but the referenced video/image files
   are openable (as-is, or after SLEAP relocates them via `search_paths` basename
   matching). Fix = load labels then `labels.save(out.pkg.slp, with_images=True)` to
   bake pixels in. Requires the referenced files to be readable at save time.
3. **`none`** — no embedded pixels, `data_path` gone/also-non-embedded, and referenced
   videos unreachable. No pixel source left. **Reported (with the paths searched) and
   skipped** for manual attention.

The ranking picks the least-destructive, least-lossy repair available per artifact. For
an artifact that is *not* broken (already embedded), `recoverable_via` is `already_ok`
(no repair needed). So the full value set is
`already_ok` / `already_embedded` / `referenced_videos` / `none`.

## Audit report — `audit_registry(...) -> pd.DataFrame`

One row per audited version (default: each collection's `latest`):

| column | meaning |
|---|---|
| `collection`, `version`, `is_latest` | identity |
| `size_mb` | fast triage (huge ⇒ likely embedded; KB/few-MB ⇒ likely not) — non-authoritative |
| `embedded` | **authoritative** — `has_embedded_images(downloaded_file)` |
| `n_user_frames`, `n_videos`, `n_videos_missing_pixels` | detail from `inspect_package` |
| `data_path`, `data_path_exists`, `data_path_embedded` | tier-1 recoverability |
| `referenced_recoverable` | tier-2: referenced videos openable (as-is or after remap) |
| `recoverable_via` | `already_ok` / `already_embedded` / `referenced_videos` / `none` |
| `notes` | paths searched, remap applied, etc. |

Output: printed summary **and** a CSV saved under the experiment folder — a durable
record produced before any writes. The audit is read-only and can be run directly on
the maintainer's machine (W&B authed via `~/_netrc`, Z: mounted, `sleap_v1.4.1`
present).

## Repair flow — `repair_artifact(collection, *, dry_run=True, search_paths=None, video_search=None)`

1. **Tier 1** (`data_path_embedded`): `fixed_path = data_path` on Z: — no re-embedding.
2. **Tier 2** (`referenced_videos`): download the artifact `.slp`, load it with SLEAP's
   native relocation — `sleap.load_file(downloaded_slp, search_paths=search_paths)` (or a
   `video_search` callable for arbitrary matching) so missing videos are found by
   basename — then `labels.save(fixed_path, with_images=True)`.
3. **Post-condition guard**: assert `has_embedded_images(fixed_path)` is `True`; abort
   the repair if re-embedding did not take.
4. **Re-register**: call `make_dataset_artifact(artifact_name=<same>,
   dataset_path=fixed_path, link_to_registry=True, ...)`, adding
   `metadata["images_embedded"]=True`, `metadata["repaired_from"]=<old version>`, and a
   `embedded-images-repair` tag → new version in the **same collection**, becoming
   `latest`.
5. **`dry_run=True` by default**: prints the plan (tier, `fixed_path`, target) and
   writes nothing. `--apply` performs the writes.

Because re-registration flows through `make_dataset_artifact`, the guardrail re-checks
embedding at upload time — a repair can never push a still-broken file.

## Guardrail — `make_dataset_artifact(..., require_embedded_images=True)`

- The check runs **before `wandb.init`** (fail fast, no orphan run).
- `has_embedded_images(dataset_path)` `False` + `require_embedded_images` → raise
  `ValueError` with a clear message (save with `with_images=True`, or pass
  `require_embedded_images=False`).
- `require_embedded_images=False` → log a warning and proceed (deliberate non-embedded
  uploads).
- Default `True` protects existing callers (e.g. `1_register_dataset.py`) automatically.

## Testing (`tests/test_datasets.py` + `tests/fixtures.py`)

The existing `tests/data/*.pkg.slp` are 30-byte **mock text files** ("Mock SLEAP
file: ..."), useless for real detection. Fixtures instead generate *real* tiny packages
via sleap at test time (a small numpy-backed video + 1–2 labeled frames), saved once
`with_images=True` (embedded) and once `with_images=False` (non-embedded). These tests
**skip when sleap is not importable**, so they run under `ci.yml`'s full install rather
than the lightweight `test-imports.yml` job.

| Test target | Approach |
|---|---|
| `has_embedded_images` | real fixtures → `True` (embedded), `False` (non-embedded), `False` (zero user frames) |
| `inspect_package` | fixtures → correct per-video embedded flags + `recoverable_via` |
| guardrail in `make_dataset_artifact` | mock wandb; raises on non-embedded, proceeds on embedded, warns when `require_embedded_images=False` |
| `audit_registry` | mock `api.artifact_collections`/download + `inspect_package`; assert schema + tier logic |
| `repair_artifact` | mock download/re-register; assert dry-run writes nothing, tier selection, post-condition abort |

## Experiment CLI — `experiments/2026-07-01-fix-registry-embedded-images/`

- `audit.py` — run `audit_registry`, print summary, save CSV. Read-only.
- `repair.py` — `--collection <name|all>`, dry-run default, `--apply` to write, optional
  `--search-paths <dir> [<dir> ...]` (forwarded to SLEAP's `search_paths`).
- `README.md` — usage + the tier explanation.

## Rollout

1. Land library functions + guardrail + tests (PR, green CI).
2. Run `audit.py` (read-only) → review the CSV together; confirm which collections are
   broken and their tier.
3. Run `repair.py` dry-run per broken collection → review the plan.
4. Run `repair.py --apply` with explicit go-ahead → new `latest` versions.
5. Re-audit to confirm all collections report `embedded == True`.

## Risks / open items

- **Tier-2 relocation**: if referenced videos can't be found under the supplied
  `search_paths` (basename matching), the artifact falls to `none`. The audit surfaces
  the referenced paths so appropriate `search_paths` (or a `video_search` callable) can
  be supplied. Basename collisions (same filename in multiple dirs) resolve to the first
  hit — order `search_paths` accordingly.
- **Download volume**: a full audit downloads each `latest` (soybean_lateral alone is
  ~242 MB). Acceptable for a maintainer-run, one-off audit; uses the W&B cache.
- **Registry API version-coupling**: enumeration is pinned to wandb 0.18.x semantics; a
  future wandb upgrade to 0.19+ would switch to `api.registries()`. Documented here.
