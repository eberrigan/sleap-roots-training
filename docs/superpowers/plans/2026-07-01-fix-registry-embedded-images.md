# Fix Missing Embedded Images in `sleap-roots-labels` Registry — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Repair W&B `sleap-roots-labels` registry packages that lack embedded images so they are trainable again, and add a guardrail so non-embedded packages can never be registered in the first place.

**Architecture:** Add reusable, tested functions to `sleap_roots_training/datasets.py` — a detection primitive (`has_embedded_images`), a per-file inspector (`inspect_package`), a registry auditor (`audit_registry`), and a re-embed-and-re-register repairer (`repair_artifact`) — plus a `require_embedded_images` guardrail on the existing `make_dataset_artifact`. A thin CLI under `experiments/2026-07-01-fix-registry-embedded-images/` drives audit (read-only) and gated repair.

**Tech Stack:** Python 3.8, `sleap` 1.4.1 (imported lazily), `wandb` 0.18.7, `pandas`, `pytest`, `black`.

## Global Constraints

- **Python 3.8 compatible** — use `typing.Optional`/`List`/`Dict`/`Any`, never `list[str]`/`X | Y` syntax.
- **`sleap` is imported lazily inside functions**, never at `datasets.py` module top-level — the module must stay importable without SLEAP (the cross-platform `test-imports.yml` job has no SLEAP).
- **Detection is scoped to videos that carry user-labeled frames** — a package can legitimately contain thousands of videos with zero embedded frames; only videos with user labels must be embedded.
- **Repairs embed with `labels.save(path, with_images=True)` only** (not `embed_all_labeled`/`embed_suggested`).
- **Fixed packages land as a new version in the same collection** (W&B artifacts are immutable).
- **Registry enumeration uses wandb 0.18.7 semantics**: `api.artifact_collections("<entity>-org/wandb-registry-<registry>", "dataset")`, then `collection.artifacts()`. `api.registries()` does NOT exist in 0.18.7.
- **Filepath remapping uses SLEAP's native `sleap.load_file(..., search_paths=[...])`** (basename matching via `find_path_using_paths`), not hand-rolled prefix maps.
- **Tests that build/read real SLEAP packages must skip when SLEAP is unavailable** via `pytest.importorskip("sleap")` inside the fixture. Mock-only tests need no SLEAP.
- **Run tests in the `sleap_v1.4.1` conda env**, e.g. `conda run -n sleap_v1.4.1 --no-capture-output python -m pytest ...`.
- **Format every touched Python file with `python -m black` before committing.**

---

## File Structure

- **Modify** `sleap_roots_training/datasets.py` — add `_video_has_embedded`, `has_embedded_images`, `_referenced_paths`, `inspect_package`, `_latest_version`, `_find_slp`, `_classify_recoverability`, `_audit_one_artifact`, `audit_registry`, `repair_artifact`; extend `make_dataset_artifact` with `require_embedded_images` + `metadata` params + guardrail.
- **Modify** `tests/fixtures.py` — add `embedded_package` and `nonembedded_package` fixtures (build real tiny packages).
- **Modify** `tests/test_datasets.py` — add tests for all new functions; add an autouse guard-bypass fixture to the existing `TestMakeDatasetArtifact` class.
- **Create** `experiments/2026-07-01-fix-registry-embedded-images/audit.py` — read-only audit CLI.
- **Create** `experiments/2026-07-01-fix-registry-embedded-images/repair.py` — gated repair CLI.
- **Create** `experiments/2026-07-01-fix-registry-embedded-images/README.md` — usage + tiers.

---

## Task 1: Detection primitive `has_embedded_images` + real package fixtures

**Files:**
- Modify: `sleap_roots_training/datasets.py` (add imports, `_video_has_embedded`, `has_embedded_images`)
- Modify: `tests/fixtures.py` (add `embedded_package`, `nonembedded_package` fixtures)
- Test: `tests/test_datasets.py` (add `TestHasEmbeddedImages`)

**Interfaces:**
- Produces:
  - `_video_has_embedded(backend) -> bool` — True iff `backend` is an `HDF5Video` whose `has_embedded_images` is truthy; never raises.
  - `has_embedded_images(path: str) -> bool` — True iff the package at `path` has embedded pixels for every video that carries user-labeled frames, and has ≥1 user-labeled frame; returns False on any load error.
  - Fixtures `embedded_package(tmp_path) -> str` and `nonembedded_package(tmp_path) -> str` returning `.slp` paths.

- [ ] **Step 1: Add the fixtures to `tests/fixtures.py`**

First add `import numpy as np` to the import block at the **top** of `tests/fixtures.py` (below the existing `import shutil`). Then append the helper and fixtures to the **end** of the file:

```python
def _build_tiny_labels(img_dir):
    """Build a 3-frame, 2-node SLEAP Labels backed by real PNG files.

    Uses image files (not numpy) so the embedded package's source video is a lazy
    SingleImageVideo — a numpy source makes HDF5Video.has_embedded_images raise on load.
    """
    import os
    import imageio.v2 as imageio
    import sleap

    os.makedirs(img_dir, exist_ok=True)
    paths = []
    for i in range(3):
        p = os.path.join(img_dir, f"frame_{i}.png")
        imageio.imwrite(p, np.random.randint(0, 255, size=(16, 16), dtype=np.uint8))
        paths.append(p)

    video = sleap.Video.from_image_filenames(paths)
    skeleton = sleap.Skeleton()
    skeleton.add_node("a")
    skeleton.add_node("b")
    frames = [
        sleap.LabeledFrame(
            video=video,
            frame_idx=i,
            instances=[
                sleap.Instance.from_pointsarray(
                    np.array([[1 + i, 1 + i], [2 + i, 2 + i]]), skeleton=skeleton
                )
            ],
        )
        for i in range(2)
    ]
    return sleap.Labels(frames), paths


@pytest.fixture
def embedded_package(tmp_path):
    """Path to a tiny .pkg.slp saved WITH embedded images."""
    pytest.importorskip("sleap")
    pytest.importorskip("imageio")
    labels, _ = _build_tiny_labels(str(tmp_path / "imgs"))
    out = str(tmp_path / "embedded.pkg.slp")
    labels.save(out, with_images=True)
    return out


@pytest.fixture
def nonembedded_package(tmp_path):
    """Path to a tiny .slp saved WITHOUT embedded images (references PNGs that exist)."""
    pytest.importorskip("sleap")
    pytest.importorskip("imageio")
    labels, img_paths = _build_tiny_labels(str(tmp_path / "imgs"))
    out = str(tmp_path / "nonembedded.slp")
    labels.save(out, with_images=False)
    return out
```

- [ ] **Step 2: Write the failing tests**

Append to `tests/test_datasets.py`:

```python
from unittest.mock import patch
from types import SimpleNamespace

from tests.fixtures import embedded_package, nonembedded_package


class TestHasEmbeddedImages:
    """Tests for has_embedded_images detection."""

    def test_embedded_package_returns_true(self, embedded_package):
        from sleap_roots_training.datasets import has_embedded_images

        assert has_embedded_images(embedded_package) is True

    def test_nonembedded_package_returns_false(self, nonembedded_package):
        from sleap_roots_training.datasets import has_embedded_images

        assert has_embedded_images(nonembedded_package) is False

    def test_zero_user_frames_returns_false(self):
        from sleap_roots_training.datasets import has_embedded_images

        fake_labels = SimpleNamespace(labeled_frames=[], videos=[])
        fake_sleap = SimpleNamespace(load_file=lambda *a, **k: fake_labels)
        # Patch the module-level sleap sentinel so _get_sleap() returns our fake and no
        # real sleap import is needed (this test runs even without sleap installed).
        with patch("sleap_roots_training.datasets.sleap", fake_sleap):
            assert has_embedded_images("whatever.slp") is False

    def test_unloadable_file_returns_false(self):
        from sleap_roots_training.datasets import has_embedded_images

        def boom(*a, **k):
            raise ValueError("bad file")

        fake_sleap = SimpleNamespace(load_file=boom)
        with patch("sleap_roots_training.datasets.sleap", fake_sleap):
            assert has_embedded_images("whatever.slp") is False
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `conda run -n sleap_v1.4.1 --no-capture-output python -m pytest tests/test_datasets.py::TestHasEmbeddedImages -v`
Expected: FAIL with `ImportError`/`AttributeError` (`has_embedded_images` does not exist).

- [ ] **Step 4: Implement the functions**

In `sleap_roots_training/datasets.py`, the existing top imports are:

```python
import wandb
import logging

from pathlib import Path
from typing import List, Optional
```

Change them to (add `import os`, widen the typing import, and add a lazy `sleap` sentinel):

```python
import os
import wandb
import logging

from pathlib import Path
from typing import Any, Dict, List, Optional

# sleap is heavy and optional at import time (the cross-platform test-imports job has no
# sleap). Keep it out of module import; _get_sleap() imports it lazily on first use and
# caches it here. Tests patch `sleap_roots_training.datasets.sleap` directly.
sleap = None
```

Then add these functions **below the existing imports/`logging.basicConfig` line and above `make_dataset_artifact`**:

```python
def _get_sleap():
    """Import and cache the ``sleap`` module lazily (kept out of module import time)."""
    global sleap
    if sleap is None:
        import sleap as _sleap

        sleap = _sleap
    return sleap


def _video_has_embedded(backend) -> bool:
    """Return True iff a video backend is an HDF5Video carrying embedded images.

    Never raises: any error accessing the backend (e.g. an unreadable source video)
    is treated as "not embedded".
    """
    try:
        return type(backend).__name__ == "HDF5Video" and bool(
            backend.has_embedded_images
        )
    except Exception:
        return False


def has_embedded_images(path: str) -> bool:
    """Return True iff the SLEAP package at ``path`` is trainable on its own.

    A package is trainable when it has at least one user-labeled frame and every video
    that carries user-labeled frames has embedded image data. Videos without user labels
    are ignored (packages may reference thousands of unused videos). Returns False if the
    file cannot be read as a SLEAP package.

    Args:
        path: Path to a ``.slp``/``.pkg.slp`` file.

    Returns:
        True if every user-labeled-frame video has embedded images, else False.
    """
    _sleap = _get_sleap()
    try:
        labels = _sleap.load_file(path)
    except Exception as e:
        logging.debug(f"has_embedded_images: could not load {path}: {e}")
        return False

    videos_with_user_frames = {
        id(lf.video) for lf in labels.labeled_frames if lf.has_user_instances
    }
    if not videos_with_user_frames:
        return False

    for video in labels.videos:
        if id(video) not in videos_with_user_frames:
            continue
        if not _video_has_embedded(video.backend):
            return False
    return True
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `conda run -n sleap_v1.4.1 --no-capture-output python -m pytest tests/test_datasets.py::TestHasEmbeddedImages -v`
Expected: PASS (4 passed).

- [ ] **Step 6: Format and commit**

```bash
python -m black sleap_roots_training/datasets.py tests/fixtures.py tests/test_datasets.py
git add sleap_roots_training/datasets.py tests/fixtures.py tests/test_datasets.py
git commit -m "feat(datasets): add has_embedded_images detection + real package fixtures"
```

---

## Task 2: Per-file inspector `inspect_package`

**Files:**
- Modify: `sleap_roots_training/datasets.py` (add `_referenced_paths`, `inspect_package`)
- Test: `tests/test_datasets.py` (add `TestInspectPackage`)

**Interfaces:**
- Consumes: `_video_has_embedded` (Task 1).
- Produces: `inspect_package(path: str, search_paths: Optional[List[str]] = None) -> Dict[str, Any]` returning keys:
  `embedded: bool`, `loadable: bool`, `error: Optional[str]`, `n_user_frames: int`, `n_videos: int`,
  `n_videos_missing_pixels: int`, `recoverable_via: str` (`"already_ok"|"referenced_videos"|"none"`),
  `referenced_paths: List[str]`, `videos: List[Dict[str, Any]]`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_datasets.py`:

```python
class TestInspectPackage:
    """Tests for inspect_package."""

    def test_embedded_package_report(self, embedded_package):
        from sleap_roots_training.datasets import inspect_package

        info = inspect_package(embedded_package)
        assert info["loadable"] is True
        assert info["embedded"] is True
        assert info["n_user_frames"] == 2
        assert info["n_videos_missing_pixels"] == 0
        assert info["recoverable_via"] == "already_ok"

    def test_nonembedded_recoverable_via_referenced_videos(self, nonembedded_package):
        from sleap_roots_training.datasets import inspect_package

        info = inspect_package(nonembedded_package)
        assert info["embedded"] is False
        assert info["n_videos_missing_pixels"] == 1
        # referenced PNGs still exist next to the fixture -> recoverable by re-embedding
        assert info["recoverable_via"] == "referenced_videos"
        assert len(info["referenced_paths"]) >= 1

    def test_unloadable_file_report(self):
        from sleap_roots_training.datasets import inspect_package

        def boom(*a, **k):
            raise ValueError("bad")

        fake_sleap = SimpleNamespace(load_file=boom)
        with patch("sleap_roots_training.datasets.sleap", fake_sleap):
            info = inspect_package("whatever.slp")
        assert info["loadable"] is False
        assert info["embedded"] is False
        assert info["recoverable_via"] == "none"
        assert info["error"]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `conda run -n sleap_v1.4.1 --no-capture-output python -m pytest tests/test_datasets.py::TestInspectPackage -v`
Expected: FAIL (`inspect_package` does not exist).

- [ ] **Step 3: Implement the functions**

Add to `sleap_roots_training/datasets.py` below `has_embedded_images`:

```python
def _referenced_paths(backend) -> List[str]:
    """Return the external file path(s) a video backend references, if any."""
    filenames = getattr(backend, "filenames", None)
    if filenames:
        return [str(f) for f in filenames]
    filename = getattr(backend, "filename", None)
    return [str(filename)] if filename else []


def inspect_package(
    path: str, search_paths: Optional[List[str]] = None
) -> Dict[str, Any]:
    """Inspect a SLEAP package's embedding status and recoverability.

    Args:
        path: Path to a ``.slp``/``.pkg.slp`` file.
        search_paths: Optional directories forwarded to ``sleap.load_file`` to relocate
            missing referenced videos by basename before checking their existence.

    Returns:
        A dict describing embedding status, per-video detail, and how (if at all) a
        broken package could be repaired from its own referenced videos:
        ``recoverable_via`` is ``"already_ok"`` (fully embedded), ``"referenced_videos"``
        (re-embeddable from reachable source files), or ``"none"``.
    """
    _sleap = _get_sleap()

    result: Dict[str, Any] = {
        "embedded": False,
        "loadable": False,
        "error": None,
        "n_user_frames": 0,
        "n_videos": 0,
        "n_videos_missing_pixels": 0,
        "recoverable_via": "none",
        "referenced_paths": [],
        "videos": [],
    }

    try:
        if search_paths:
            labels = _sleap.load_file(path, search_paths=search_paths)
        else:
            labels = _sleap.load_file(path)
    except Exception as e:
        result["error"] = f"{type(e).__name__}: {e}"
        return result

    result["loadable"] = True
    videos_with_user_frames = {
        id(lf.video) for lf in labels.labeled_frames if lf.has_user_instances
    }
    result["n_user_frames"] = sum(
        1 for lf in labels.labeled_frames if lf.has_user_instances
    )
    result["n_videos"] = len(labels.videos)

    missing_referenced: List[str] = []
    all_missing_reachable = True
    n_missing = 0

    for video in labels.videos:
        backend = video.backend
        has_user = id(video) in videos_with_user_frames
        embedded = _video_has_embedded(backend)
        refs = _referenced_paths(backend)
        refs_exist = all(os.path.exists(p) for p in refs) if refs else False
        result["videos"].append(
            {
                "backend_type": type(backend).__name__,
                "embedded": embedded,
                "has_user_frames": has_user,
                "referenced_paths": refs,
                "referenced_exists": refs_exist,
            }
        )
        if has_user and not embedded:
            n_missing += 1
            missing_referenced.extend(refs)
            if not refs_exist:
                all_missing_reachable = False

    result["n_videos_missing_pixels"] = n_missing
    result["referenced_paths"] = missing_referenced
    result["embedded"] = result["n_user_frames"] > 0 and n_missing == 0

    if result["embedded"]:
        result["recoverable_via"] = "already_ok"
    elif n_missing > 0 and all_missing_reachable:
        result["recoverable_via"] = "referenced_videos"
    else:
        result["recoverable_via"] = "none"

    return result
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `conda run -n sleap_v1.4.1 --no-capture-output python -m pytest tests/test_datasets.py::TestInspectPackage -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Format and commit**

```bash
python -m black sleap_roots_training/datasets.py tests/test_datasets.py
git add sleap_roots_training/datasets.py tests/test_datasets.py
git commit -m "feat(datasets): add inspect_package with per-video recoverability"
```

---

## Task 3: Guardrail + `metadata` param on `make_dataset_artifact`

**Files:**
- Modify: `sleap_roots_training/datasets.py` (`make_dataset_artifact` signature + guardrail + metadata merge)
- Test: `tests/test_datasets.py` (autouse bypass fixture on `TestMakeDatasetArtifact`; new `TestEmbeddingGuardrail`)

**Interfaces:**
- Consumes: `has_embedded_images` (Task 1).
- Produces: `make_dataset_artifact(..., require_embedded_images: bool = True, metadata: Optional[Dict[str, Any]] = None)` — raises `ValueError` before `wandb.init` when the file lacks embedded images and `require_embedded_images` is True; merges `metadata` into `artifact.metadata`.

- [ ] **Step 1: Add an autouse bypass fixture so existing plumbing tests keep passing**

The existing `TestMakeDatasetArtifact` tests pass tiny **text** `.slp` files, which the new guardrail would reject. Add this autouse fixture as the **first member** of the `class TestMakeDatasetArtifact:` block in `tests/test_datasets.py` (it patches the module-level `has_embedded_images` to True for every test in that class only):

```python
    @pytest.fixture(autouse=True)
    def _bypass_embedding_guard(self):
        with patch(
            "sleap_roots_training.datasets.has_embedded_images", return_value=True
        ):
            yield
```

Ensure `import pytest` and `from unittest.mock import patch` are present at the top of `tests/test_datasets.py` (they already are).

- [ ] **Step 2: Write the failing guardrail tests**

Append to `tests/test_datasets.py`:

```python
import tempfile


class TestEmbeddingGuardrail:
    """Tests for the require_embedded_images guardrail in make_dataset_artifact."""

    @patch("sleap_roots_training.datasets.wandb.init")
    @patch("sleap_roots_training.datasets.CONFIG")
    @patch("sleap_roots_training.datasets.has_embedded_images", return_value=False)
    def test_raises_on_nonembedded_by_default(
        self, mock_embed, mock_config, mock_wandb_init
    ):
        from sleap_roots_training.datasets import make_dataset_artifact

        mock_config.__getitem__.side_effect = lambda k: "x"
        with pytest.raises(ValueError, match="no embedded images"):
            make_dataset_artifact(
                artifact_name="a", dataset_path="/tmp/broken.slp"
            )
        # Guardrail runs before wandb.init -> no orphan run.
        mock_wandb_init.assert_not_called()

    @patch("sleap_roots_training.datasets.wandb.init")
    @patch("sleap_roots_training.datasets.CONFIG")
    @patch("sleap_roots_training.datasets.has_embedded_images", return_value=False)
    def test_warns_and_proceeds_when_disabled(
        self, mock_embed, mock_config, mock_wandb_init
    ):
        from sleap_roots_training.datasets import make_dataset_artifact

        mock_config.__getitem__.side_effect = lambda key: {
            "project_name": "p",
            "entity_name": "e",
            "experiment_name": "x",
            "registry": "r",
            "collection_name": "c",
        }[key]
        mock_run = MagicMock()
        mock_wandb_init.return_value = mock_run
        mock_artifact = MagicMock()
        mock_artifact.metadata = {}
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "broken.slp"
            p.write_text("x")
            with patch(
                "sleap_roots_training.datasets.wandb.Artifact",
                return_value=mock_artifact,
            ):
                result = make_dataset_artifact(
                    artifact_name="a",
                    dataset_path=str(p),
                    require_embedded_images=False,
                )
        mock_wandb_init.assert_called_once()
        assert result == mock_artifact

    @patch("sleap_roots_training.datasets.wandb.init")
    @patch("sleap_roots_training.datasets.CONFIG")
    @patch("sleap_roots_training.datasets.has_embedded_images", return_value=True)
    def test_merges_metadata(self, mock_embed, mock_config, mock_wandb_init):
        from sleap_roots_training.datasets import make_dataset_artifact

        mock_config.__getitem__.side_effect = lambda key: {
            "project_name": "p",
            "entity_name": "e",
            "experiment_name": "x",
            "registry": "r",
            "collection_name": "c",
        }[key]
        mock_run = MagicMock()
        mock_wandb_init.return_value = mock_run
        mock_artifact = MagicMock()
        mock_artifact.metadata = {}
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "ok.pkg.slp"
            p.write_text("x")
            with patch(
                "sleap_roots_training.datasets.wandb.Artifact",
                return_value=mock_artifact,
            ):
                make_dataset_artifact(
                    artifact_name="a",
                    dataset_path=str(p),
                    metadata={"images_embedded": True, "repaired_from": "v0"},
                )
        assert mock_artifact.metadata["images_embedded"] is True
        assert mock_artifact.metadata["repaired_from"] == "v0"
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `conda run -n sleap_v1.4.1 --no-capture-output python -m pytest tests/test_datasets.py::TestEmbeddingGuardrail -v`
Expected: FAIL (`make_dataset_artifact` has no `require_embedded_images`/`metadata` params; no guardrail).

- [ ] **Step 4: Implement the guardrail + metadata merge**

In `sleap_roots_training/datasets.py`, change the `make_dataset_artifact` signature from:

```python
def make_dataset_artifact(
    artifact_name: str,
    dataset_path: str,
    link_to_registry: bool = False,
    description: Optional[str] = None,
    tags: Optional[List[str]] = None,
) -> wandb.Artifact:
```

to:

```python
def make_dataset_artifact(
    artifact_name: str,
    dataset_path: str,
    link_to_registry: bool = False,
    description: Optional[str] = None,
    tags: Optional[List[str]] = None,
    require_embedded_images: bool = True,
    metadata: Optional[Dict[str, Any]] = None,
) -> wandb.Artifact:
```

Immediately after the CONFIG reads (the block ending with `COLLECTION_NAME = CONFIG["collection_name"]`) and **before** `run = wandb.init(`, insert:

```python
    # Guardrail: refuse to register a package that lacks embedded images, since it
    # cannot be used for remote training. Runs before wandb.init to avoid orphan runs.
    dataset_path = Path(dataset_path)
    if not has_embedded_images(dataset_path.as_posix()):
        message = (
            f"Refusing to register '{dataset_path.as_posix()}': it has no embedded "
            "images (or could not be read as a SLEAP package). Save it with "
            "`labels.save(path, with_images=True)`, or pass "
            "`require_embedded_images=False` to register anyway."
        )
        if require_embedded_images:
            raise ValueError(message)
        logging.warning(message)
```

Then inside the `try:` block, **delete** the now-duplicate line:

```python
        dataset_path = Path(dataset_path)
```

Finally, right after the existing tags-to-metadata loop (the block:
```python
        if tags:
            for tag in tags:
                artifact.metadata[tag] = True
```
) add:

```python
        if metadata:
            artifact.metadata.update(metadata)
```

- [ ] **Step 5: Run the full datasets test module to verify everything passes**

Run: `conda run -n sleap_v1.4.1 --no-capture-output python -m pytest tests/test_datasets.py -v`
Expected: PASS (existing `TestMakeDatasetArtifact` tests + new guardrail tests all green).

- [ ] **Step 6: Format and commit**

```bash
python -m black sleap_roots_training/datasets.py tests/test_datasets.py
git add sleap_roots_training/datasets.py tests/test_datasets.py
git commit -m "feat(datasets): add require_embedded_images guardrail + metadata to make_dataset_artifact"
```

---

## Task 4: Recoverability classifier + version/file helpers

**Files:**
- Modify: `sleap_roots_training/datasets.py` (add `_classify_recoverability`, `_latest_version`, `_find_slp`)
- Test: `tests/test_datasets.py` (add `TestRecoverabilityHelpers`)

**Interfaces:**
- Produces:
  - `_classify_recoverability(info: Dict[str, Any], data_path_embedded: bool) -> str` → `"already_ok"|"already_embedded"|"referenced_videos"|"none"`.
  - `_latest_version(versions: List) -> Optional[object]` — the version whose `aliases` contains `"latest"`, else the first, else None.
  - `_find_slp(directory: str) -> Optional[str]` — path to the first `*.slp`/`*.pkg.slp` under `directory`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_datasets.py`:

```python
class TestRecoverabilityHelpers:
    """Tests for _classify_recoverability, _latest_version, _find_slp."""

    def test_classify_already_ok(self):
        from sleap_roots_training.datasets import _classify_recoverability

        assert (
            _classify_recoverability({"embedded": True, "recoverable_via": "already_ok"}, False)
            == "already_ok"
        )

    def test_classify_already_embedded_beats_referenced(self):
        from sleap_roots_training.datasets import _classify_recoverability

        info = {"embedded": False, "recoverable_via": "referenced_videos"}
        assert _classify_recoverability(info, data_path_embedded=True) == "already_embedded"

    def test_classify_referenced_videos(self):
        from sleap_roots_training.datasets import _classify_recoverability

        info = {"embedded": False, "recoverable_via": "referenced_videos"}
        assert _classify_recoverability(info, data_path_embedded=False) == "referenced_videos"

    def test_classify_none(self):
        from sleap_roots_training.datasets import _classify_recoverability

        info = {"embedded": False, "recoverable_via": "none"}
        assert _classify_recoverability(info, data_path_embedded=False) == "none"

    def test_latest_version_prefers_latest_alias(self):
        from sleap_roots_training.datasets import _latest_version

        v0 = SimpleNamespace(aliases=[], version="v0")
        v1 = SimpleNamespace(aliases=["latest"], version="v1")
        assert _latest_version([v0, v1]) is v1

    def test_latest_version_falls_back_to_first(self):
        from sleap_roots_training.datasets import _latest_version

        v0 = SimpleNamespace(aliases=[], version="v0")
        assert _latest_version([v0]) is v0
        assert _latest_version([]) is None

    def test_find_slp(self, tmp_path):
        from sleap_roots_training.datasets import _find_slp

        (tmp_path / "notes.txt").write_text("x")
        slp = tmp_path / "labels.pkg.slp"
        slp.write_text("x")
        assert _find_slp(str(tmp_path)) == str(slp)
        assert _find_slp(str(tmp_path / "empty")) is None
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `conda run -n sleap_v1.4.1 --no-capture-output python -m pytest tests/test_datasets.py::TestRecoverabilityHelpers -v`
Expected: FAIL (helpers do not exist).

- [ ] **Step 3: Implement the helpers**

Add to `sleap_roots_training/datasets.py` below `inspect_package`:

```python
def _classify_recoverability(info: Dict[str, Any], data_path_embedded: bool) -> str:
    """Combine per-file inspection with the registry ``data_path`` tier-1 check.

    Returns one of ``already_ok`` (artifact already embedded), ``already_embedded``
    (the metadata data_path file on disk is embedded), ``referenced_videos``
    (re-embeddable from reachable source videos), or ``none``.
    """
    if info.get("embedded"):
        return "already_ok"
    if data_path_embedded:
        return "already_embedded"
    if info.get("recoverable_via") == "referenced_videos":
        return "referenced_videos"
    return "none"


def _latest_version(versions: List) -> Optional[object]:
    """Return the version tagged ``latest``, else the first, else None."""
    for v in versions:
        if "latest" in (getattr(v, "aliases", None) or []):
            return v
    return versions[0] if versions else None


def _find_slp(directory: str) -> Optional[str]:
    """Return the first ``.slp``/``.pkg.slp`` file under ``directory`` (sorted)."""
    import glob
    import os

    matches = sorted(glob.glob(os.path.join(directory, "**", "*.slp"), recursive=True))
    return matches[0] if matches else None
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `conda run -n sleap_v1.4.1 --no-capture-output python -m pytest tests/test_datasets.py::TestRecoverabilityHelpers -v`
Expected: PASS (7 passed).

- [ ] **Step 5: Format and commit**

```bash
python -m black sleap_roots_training/datasets.py tests/test_datasets.py
git add sleap_roots_training/datasets.py tests/test_datasets.py
git commit -m "feat(datasets): add recoverability classifier + version/slp helpers"
```

---

## Task 5: `audit_registry`

**Files:**
- Modify: `sleap_roots_training/datasets.py` (add `_audit_one_artifact`, `audit_registry`)
- Test: `tests/test_datasets.py` (add `TestAuditRegistry`)

**Interfaces:**
- Consumes: `inspect_package`, `has_embedded_images`, `_classify_recoverability`, `_latest_version`, `_find_slp`.
- Produces:
  - `_audit_one_artifact(collection_name: str, artifact, download_root: Optional[str], search_paths: Optional[List[str]]) -> Dict[str, Any]` — one report row.
  - `audit_registry(registry=None, entity=None, collections=None, all_versions=False, download_root=None, search_paths=None) -> pandas.DataFrame` with columns: `collection, version, is_latest, size_mb, embedded, n_user_frames, n_videos, n_videos_missing_pixels, data_path, data_path_exists, data_path_embedded, referenced_recoverable, recoverable_via, notes`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_datasets.py`:

```python
class TestAuditRegistry:
    """Tests for audit_registry orchestration (all wandb/sleap calls mocked)."""

    def _fake_artifact(self, version, aliases, metadata, size):
        art = MagicMock()
        art.version = version
        art.aliases = aliases
        art.metadata = metadata
        art.size = size
        art.download.return_value = f"/dl/{version}"
        return art

    @patch("sleap_roots_training.datasets.os.path.exists", return_value=True)
    @patch("sleap_roots_training.datasets.has_embedded_images", return_value=True)
    @patch("sleap_roots_training.datasets._find_slp", return_value="/dl/v0/labels.slp")
    @patch("sleap_roots_training.datasets.inspect_package")
    @patch("sleap_roots_training.datasets.wandb.Api")
    @patch("sleap_roots_training.datasets.CONFIG")
    def test_audit_builds_expected_row(
        self,
        mock_config,
        mock_api_cls,
        mock_inspect,
        mock_find_slp,
        mock_has_embed,
        mock_exists,
    ):
        from sleap_roots_training.datasets import audit_registry

        mock_config.__getitem__.side_effect = lambda key: {
            "entity_name": "ent",
            "registry": "sleap-roots-labels",
        }[key]

        # One collection, one latest version that is NOT embedded but data_path is.
        art = self._fake_artifact(
            "v0", ["latest"], {"data_path": "Z:/src/labels.pkg.slp"}, 5_000_000
        )
        coll = MagicMock()
        coll.name = "soybean_primary_6nodes_v004_labels"
        coll.artifacts.return_value = [art]

        api = MagicMock()
        api.artifact_collections.return_value = [coll]
        mock_api_cls.return_value = api

        mock_inspect.return_value = {
            "embedded": False,
            "n_user_frames": 10,
            "n_videos": 3,
            "n_videos_missing_pixels": 1,
            "recoverable_via": "none",
            "error": None,
        }

        df = audit_registry()

        api.artifact_collections.assert_called_once_with(
            "ent-org/wandb-registry-sleap-roots-labels", "dataset"
        )
        assert len(df) == 1
        row = df.iloc[0]
        assert row["collection"] == "soybean_primary_6nodes_v004_labels"
        assert row["version"] == "v0"
        assert row["is_latest"] is True
        assert row["embedded"] is False
        # data_path exists + has_embedded_images True -> tier already_embedded
        assert row["data_path_embedded"] is True
        assert row["recoverable_via"] == "already_embedded"
        assert set(
            [
                "collection",
                "version",
                "is_latest",
                "size_mb",
                "embedded",
                "n_user_frames",
                "n_videos",
                "n_videos_missing_pixels",
                "data_path",
                "data_path_exists",
                "data_path_embedded",
                "referenced_recoverable",
                "recoverable_via",
                "notes",
            ]
        ).issubset(df.columns)

    @patch("sleap_roots_training.datasets.wandb.Api")
    @patch("sleap_roots_training.datasets.CONFIG")
    def test_audit_filters_by_collection(self, mock_config, mock_api_cls):
        from sleap_roots_training.datasets import audit_registry

        mock_config.__getitem__.side_effect = lambda key: {
            "entity_name": "ent",
            "registry": "sleap-roots-labels",
        }[key]
        coll = MagicMock()
        coll.name = "other_collection"
        coll.artifacts.return_value = []
        api = MagicMock()
        api.artifact_collections.return_value = [coll]
        mock_api_cls.return_value = api

        df = audit_registry(collections=["not_present"])
        assert len(df) == 0
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `conda run -n sleap_v1.4.1 --no-capture-output python -m pytest tests/test_datasets.py::TestAuditRegistry -v`
Expected: FAIL (`audit_registry` does not exist).

- [ ] **Step 3: Implement `audit_registry`**

`os` is already a top-level import (added in Task 1). Add below `_find_slp`:

```python
def _audit_one_artifact(
    collection_name: str,
    artifact,
    download_root: Optional[str],
    search_paths: Optional[List[str]],
) -> Dict[str, Any]:
    """Download + inspect one registry artifact version; return one report row."""
    md = getattr(artifact, "metadata", None) or {}
    data_path = md.get("data_path")
    aliases = getattr(artifact, "aliases", None) or []

    row: Dict[str, Any] = {
        "collection": collection_name,
        "version": getattr(artifact, "version", getattr(artifact, "name", "?")),
        "is_latest": "latest" in aliases,
        "size_mb": round((getattr(artifact, "size", 0) or 0) / 1_000_000, 1),
        "data_path": data_path,
    }

    art_dir = (
        artifact.download(root=download_root)
        if download_root
        else artifact.download()
    )
    slp = _find_slp(art_dir)
    if slp is None:
        row.update(
            {
                "embedded": False,
                "n_user_frames": 0,
                "n_videos": 0,
                "n_videos_missing_pixels": 0,
                "data_path_exists": False,
                "data_path_embedded": False,
                "referenced_recoverable": False,
                "recoverable_via": "none",
                "notes": "no .slp file found in artifact",
            }
        )
        return row

    info = inspect_package(slp, search_paths=search_paths)
    data_path_exists = bool(data_path) and os.path.exists(data_path)
    data_path_embedded = data_path_exists and has_embedded_images(data_path)

    row.update(
        {
            "embedded": info["embedded"],
            "n_user_frames": info["n_user_frames"],
            "n_videos": info["n_videos"],
            "n_videos_missing_pixels": info["n_videos_missing_pixels"],
            "data_path_exists": data_path_exists,
            "data_path_embedded": data_path_embedded,
            "referenced_recoverable": info["recoverable_via"] == "referenced_videos",
            "recoverable_via": _classify_recoverability(info, data_path_embedded),
            "notes": info.get("error") or "",
        }
    )
    return row


def audit_registry(
    registry: Optional[str] = None,
    entity: Optional[str] = None,
    collections: Optional[List[str]] = None,
    all_versions: bool = False,
    download_root: Optional[str] = None,
    search_paths: Optional[List[str]] = None,
) -> "Any":
    """Audit the label-package registry for missing embedded images.

    Enumerates dataset collections under ``<entity>-org/wandb-registry-<registry>``,
    downloads each collection's ``latest`` version (or all versions), and reports
    embedding status + recoverability. Returns a ``pandas.DataFrame``.

    Args:
        registry: Registry name; defaults to ``CONFIG['registry']`` or
            ``"sleap-roots-labels"``.
        entity: W&B entity; defaults to ``CONFIG['entity_name']``.
        collections: Optional subset of collection names to audit.
        all_versions: If True, audit every version, else only ``latest``.
        download_root: Optional root dir for artifact downloads.
        search_paths: Optional dirs to relocate missing referenced videos by basename.

    Returns:
        A DataFrame with one row per audited artifact version.
    """
    import pandas as pd

    entity = entity or CONFIG["entity_name"]
    registry = registry or CONFIG["registry"] or "sleap-roots-labels"
    project_path = f"{entity}-org/wandb-registry-{registry}"

    api = wandb.Api()
    rows: List[Dict[str, Any]] = []
    for coll in api.artifact_collections(project_path, "dataset"):
        if collections and coll.name not in collections:
            continue
        versions = list(coll.artifacts())
        if all_versions:
            targets = versions
        else:
            latest = _latest_version(versions)
            targets = [latest] if latest is not None else []
        for art in targets:
            rows.append(
                _audit_one_artifact(coll.name, art, download_root, search_paths)
            )

    logging.info(
        f"Audited {len(rows)} artifact version(s) across registry '{registry}'."
    )
    return pd.DataFrame(rows)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `conda run -n sleap_v1.4.1 --no-capture-output python -m pytest tests/test_datasets.py::TestAuditRegistry -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Format and commit**

```bash
python -m black sleap_roots_training/datasets.py tests/test_datasets.py
git add sleap_roots_training/datasets.py tests/test_datasets.py
git commit -m "feat(datasets): add audit_registry over sleap-roots-labels"
```

---

## Task 6: `repair_artifact`

**Files:**
- Modify: `sleap_roots_training/datasets.py` (add `repair_artifact`)
- Test: `tests/test_datasets.py` (add `TestRepairArtifact`)

**Interfaces:**
- Consumes: `has_embedded_images`, `inspect_package`, `_latest_version`, `_find_slp`, `make_dataset_artifact`, and `sleap.load_file`/`Labels.save`.
- Produces: `repair_artifact(collection: str, *, registry=None, entity=None, dry_run=True, search_paths=None, out_dir=None, download_root=None) -> Dict[str, Any]` returning `{collection, status, tier, fixed_path, recoverable_via}` where `status` ∈ `{"already_ok","unrecoverable","dry_run","applied"}`. Raises `RuntimeError` if re-embedding fails its post-condition.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_datasets.py`:

```python
class TestRepairArtifact:
    """Tests for repair_artifact (all wandb/sleap/make_dataset_artifact mocked)."""

    def _api_with_latest(self, mock_api_cls, metadata):
        art = MagicMock()
        art.version = "v0"
        art.aliases = ["latest"]
        art.metadata = metadata
        art.download.return_value = "/dl/v0"
        coll = MagicMock()
        coll.name = "col"
        coll.artifacts.return_value = [art]
        api = MagicMock()
        api.artifact_collections.return_value = [coll]
        mock_api_cls.return_value = api
        return api, art

    @patch("sleap_roots_training.datasets.make_dataset_artifact")
    @patch("sleap_roots_training.datasets.has_embedded_images", return_value=True)
    @patch("sleap_roots_training.datasets.os.path.exists", return_value=True)
    @patch("sleap_roots_training.datasets.wandb.Api")
    @patch("sleap_roots_training.datasets.CONFIG")
    def test_tier1_dry_run_no_writes(
        self, mock_config, mock_api_cls, mock_exists, mock_embed, mock_make
    ):
        from sleap_roots_training.datasets import repair_artifact

        mock_config.__getitem__.side_effect = lambda key: {
            "entity_name": "ent",
            "registry": "sleap-roots-labels",
        }[key]
        self._api_with_latest(mock_api_cls, {"data_path": "Z:/src/labels.pkg.slp"})

        result = repair_artifact("col", dry_run=True)

        assert result["tier"] == "already_embedded"
        assert result["fixed_path"] == "Z:/src/labels.pkg.slp"
        assert result["status"] == "dry_run"
        mock_make.assert_not_called()

    @patch("sleap_roots_training.config.update_config")
    @patch("sleap_roots_training.datasets.os.makedirs")
    @patch("sleap_roots_training.datasets.make_dataset_artifact")
    @patch("sleap_roots_training.datasets.inspect_package")
    @patch("sleap_roots_training.datasets.sleap")
    @patch("sleap_roots_training.datasets.has_embedded_images")
    @patch("sleap_roots_training.datasets.os.path.exists", return_value=False)
    @patch("sleap_roots_training.datasets.wandb.Api")
    @patch("sleap_roots_training.datasets.CONFIG")
    def test_tier2_apply_reembeds_and_registers(
        self,
        mock_config,
        mock_api_cls,
        mock_exists,
        mock_embed,
        mock_sleap,
        mock_inspect,
        mock_make,
        mock_makedirs,
        mock_update_config,
    ):
        from sleap_roots_training.datasets import repair_artifact

        mock_config.__getitem__.side_effect = lambda key: {
            "entity_name": "ent",
            "registry": "sleap-roots-labels",
        }[key]
        self._api_with_latest(mock_api_cls, {"data_path": "Z:/gone.pkg.slp"})

        # tier-1 check: data_path not embedded (os.path.exists False short-circuits).
        # tier-2: artifact slp is not embedded but recoverable; after save it IS.
        mock_inspect.return_value = {"embedded": False, "recoverable_via": "referenced_videos"}
        # has_embedded_images: [tier1 data_path]=skipped(exists False), [post-condition]=True
        mock_embed.return_value = True
        mock_labels = MagicMock()
        mock_sleap.load_file.return_value = mock_labels

        with patch(
            "sleap_roots_training.datasets._find_slp", return_value="/dl/v0/labels.slp"
        ):
            result = repair_artifact(
                "col", dry_run=False, search_paths=["Z:/videos"], out_dir="/out"
            )

        mock_sleap.load_file.assert_called_once_with(
            "/dl/v0/labels.slp", search_paths=["Z:/videos"]
        )
        mock_labels.save.assert_called_once()
        assert mock_labels.save.call_args.kwargs.get("with_images") is True
        mock_make.assert_called_once()
        assert mock_make.call_args.kwargs["require_embedded_images"] is True
        assert mock_make.call_args.kwargs["metadata"]["images_embedded"] is True
        assert result["status"] == "applied"
        assert result["tier"] == "referenced_videos"

    @patch("sleap_roots_training.datasets.os.makedirs")
    @patch("sleap_roots_training.datasets.make_dataset_artifact")
    @patch("sleap_roots_training.datasets.inspect_package")
    @patch("sleap_roots_training.datasets.sleap")
    @patch("sleap_roots_training.datasets.has_embedded_images")
    @patch("sleap_roots_training.datasets.os.path.exists", return_value=False)
    @patch("sleap_roots_training.datasets.wandb.Api")
    @patch("sleap_roots_training.datasets.CONFIG")
    def test_tier2_postcondition_failure_raises(
        self,
        mock_config,
        mock_api_cls,
        mock_exists,
        mock_embed,
        mock_sleap,
        mock_inspect,
        mock_make,
        mock_makedirs,
    ):
        from sleap_roots_training.datasets import repair_artifact

        mock_config.__getitem__.side_effect = lambda key: {
            "entity_name": "ent",
            "registry": "sleap-roots-labels",
        }[key]
        self._api_with_latest(mock_api_cls, {"data_path": "Z:/gone.pkg.slp"})
        mock_inspect.return_value = {"embedded": False, "recoverable_via": "referenced_videos"}
        mock_embed.return_value = False  # re-embed silently failed
        mock_sleap.load_file.return_value = MagicMock()

        with patch(
            "sleap_roots_training.datasets._find_slp", return_value="/dl/v0/labels.slp"
        ):
            with pytest.raises(RuntimeError, match="still lacks embedded images"):
                repair_artifact("col", dry_run=False, out_dir="/out")
        mock_make.assert_not_called()

    @patch("sleap_roots_training.datasets.make_dataset_artifact")
    @patch("sleap_roots_training.datasets.inspect_package")
    @patch("sleap_roots_training.datasets.has_embedded_images", return_value=False)
    @patch("sleap_roots_training.datasets.os.path.exists", return_value=False)
    @patch("sleap_roots_training.datasets.wandb.Api")
    @patch("sleap_roots_training.datasets.CONFIG")
    def test_unrecoverable_returns_status(
        self,
        mock_config,
        mock_api_cls,
        mock_exists,
        mock_embed,
        mock_inspect,
        mock_make,
    ):
        from sleap_roots_training.datasets import repair_artifact

        mock_config.__getitem__.side_effect = lambda key: {
            "entity_name": "ent",
            "registry": "sleap-roots-labels",
        }[key]
        self._api_with_latest(mock_api_cls, {"data_path": "Z:/gone.pkg.slp"})
        mock_inspect.return_value = {"embedded": False, "recoverable_via": "none"}

        with patch(
            "sleap_roots_training.datasets._find_slp", return_value="/dl/v0/labels.slp"
        ):
            result = repair_artifact("col", dry_run=False)
        assert result["status"] == "unrecoverable"
        assert result["recoverable_via"] == "none"
        mock_make.assert_not_called()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `conda run -n sleap_v1.4.1 --no-capture-output python -m pytest tests/test_datasets.py::TestRepairArtifact -v`
Expected: FAIL (`repair_artifact` does not exist; note the `sleap` attribute patch also requires a module-level `sleap` reference — added in Step 3).

- [ ] **Step 3: Implement `repair_artifact`**

`repair_artifact` obtains sleap via `_get_sleap()` (Task 1) and uses the module-level `os` (Task 1) so tests can patch `sleap_roots_training.datasets.os.path.exists`. Add `repair_artifact` below `audit_registry`:

```python
def repair_artifact(
    collection: str,
    *,
    registry: Optional[str] = None,
    entity: Optional[str] = None,
    dry_run: bool = True,
    search_paths: Optional[List[str]] = None,
    out_dir: Optional[str] = None,
    download_root: Optional[str] = None,
) -> Dict[str, Any]:
    """Repair one registry collection's latest version so it has embedded images.

    Tier 1 (``already_embedded``): if the artifact's ``metadata['data_path']`` file on
    disk is itself embedded, re-register that file — no re-embedding. Tier 2
    (``referenced_videos``): download the artifact, relocate missing videos via
    ``search_paths``, and ``labels.save(..., with_images=True)``. A post-condition
    asserts the fixed file is embedded before re-registration, which goes through
    ``make_dataset_artifact`` (so the guardrail double-checks) as a new version in the
    same collection.

    Args:
        collection: Collection name to repair.
        registry: Registry name; defaults to ``CONFIG['registry']`` or
            ``"sleap-roots-labels"``.
        entity: W&B entity; defaults to ``CONFIG['entity_name']``.
        dry_run: If True (default), plan only — no downloads-to-registry writes.
        search_paths: Dirs to relocate missing referenced videos by basename.
        out_dir: Where to write the re-embedded package (tier 2). Defaults to a temp dir.
        download_root: Optional root dir for artifact downloads.

    Returns:
        A dict: ``{collection, status, tier, fixed_path, recoverable_via}``. ``status``
        is one of ``already_ok``, ``unrecoverable``, ``dry_run``, ``applied``.

    Raises:
        RuntimeError: If tier-2 re-embedding does not produce an embedded package.
    """
    import tempfile

    _sleap = _get_sleap()
    entity = entity or CONFIG["entity_name"]
    registry = registry or CONFIG["registry"] or "sleap-roots-labels"
    project_path = f"{entity}-org/wandb-registry-{registry}"

    api = wandb.Api()
    target_coll = None
    for coll in api.artifact_collections(project_path, "dataset"):
        if coll.name == collection:
            target_coll = coll
            break
    if target_coll is None:
        raise ValueError(f"Collection '{collection}' not found in registry '{registry}'.")

    artifact = _latest_version(list(target_coll.artifacts()))
    if artifact is None:
        raise ValueError(f"Collection '{collection}' has no versions.")

    md = getattr(artifact, "metadata", None) or {}
    data_path = md.get("data_path")
    old_version = getattr(artifact, "version", "?")

    result: Dict[str, Any] = {
        "collection": collection,
        "status": None,
        "tier": None,
        "fixed_path": None,
        "recoverable_via": None,
    }

    # Tier 1: the source file on disk is already embedded -> re-register it as-is.
    if data_path and os.path.exists(data_path) and has_embedded_images(data_path):
        tier = "already_embedded"
        fixed_path = data_path
    else:
        art_dir = (
            artifact.download(root=download_root)
            if download_root
            else artifact.download()
        )
        slp = _find_slp(art_dir)
        info = inspect_package(slp, search_paths=search_paths)
        result["recoverable_via"] = info.get("recoverable_via")

        if info.get("embedded"):
            result["status"] = "already_ok"
            result["tier"] = "already_ok"
            result["fixed_path"] = slp
            return result
        if info.get("recoverable_via") != "referenced_videos":
            result["status"] = "unrecoverable"
            return result

        tier = "referenced_videos"
        out_dir = out_dir or tempfile.mkdtemp(prefix="reembed_")
        os.makedirs(out_dir, exist_ok=True)
        fixed_path = os.path.join(out_dir, f"{collection}.pkg.slp")
        if search_paths:
            labels = _sleap.load_file(slp, search_paths=search_paths)
        else:
            labels = _sleap.load_file(slp)
        labels.save(fixed_path, with_images=True)

    # Post-condition: the fixed file must actually be embedded.
    if not has_embedded_images(fixed_path):
        raise RuntimeError(
            f"Repair failed: '{fixed_path}' still lacks embedded images after re-embedding."
        )

    result["tier"] = tier
    result["fixed_path"] = fixed_path
    result["recoverable_via"] = tier

    if dry_run:
        result["status"] = "dry_run"
        return result

    from sleap_roots_training import config as _config

    _config.update_config(
        registry=registry,
        collection_name=collection,
        experiment_name=f"{collection}_embedded_images_repair",
        job_type="build_dataset",
    )
    make_dataset_artifact(
        artifact_name=collection,
        dataset_path=fixed_path,
        link_to_registry=True,
        description=f"Re-embedded ({tier}) repair of '{collection}' to restore trainable images.",
        tags=["embedded-images-repair"],
        require_embedded_images=True,
        metadata={"images_embedded": True, "repaired_from": old_version},
    )
    result["status"] = "applied"
    return result
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `conda run -n sleap_v1.4.1 --no-capture-output python -m pytest tests/test_datasets.py::TestRepairArtifact -v`
Expected: PASS (4 passed).

- [ ] **Step 5: Run the full datasets module + confirm SLEAP-dependent tests still pass**

Run: `conda run -n sleap_v1.4.1 --no-capture-output python -m pytest tests/test_datasets.py -v`
Expected: PASS (all classes green).

- [ ] **Step 6: Format and commit**

```bash
python -m black sleap_roots_training/datasets.py tests/test_datasets.py
git add sleap_roots_training/datasets.py tests/test_datasets.py
git commit -m "feat(datasets): add repair_artifact with tiered re-embed + gated re-register"
```

---

## Task 7: Experiment CLI (audit + repair) and README

**Files:**
- Create: `experiments/2026-07-01-fix-registry-embedded-images/audit.py`
- Create: `experiments/2026-07-01-fix-registry-embedded-images/repair.py`
- Create: `experiments/2026-07-01-fix-registry-embedded-images/README.md`

**Interfaces:**
- Consumes: `audit_registry`, `repair_artifact` (Tasks 5–6).
- Produces: two runnable scripts; no library exports.

- [ ] **Step 1: Create `audit.py`**

Create `experiments/2026-07-01-fix-registry-embedded-images/audit.py`:

```python
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
```

- [ ] **Step 2: Create `repair.py`**

Create `experiments/2026-07-01-fix-registry-embedded-images/repair.py`:

```python
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
```

- [ ] **Step 3: Create `README.md`**

Create `experiments/2026-07-01-fix-registry-embedded-images/README.md`:

```markdown
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

After applying, re-run `audit.py` to confirm every collection reports `embedded == True`.
```

- [ ] **Step 4: Verify the CLIs parse (smoke test)**

Run: `conda run -n sleap_v1.4.1 --no-capture-output python experiments/2026-07-01-fix-registry-embedded-images/audit.py --help`
Expected: argparse help text prints, exit 0.

Run: `conda run -n sleap_v1.4.1 --no-capture-output python experiments/2026-07-01-fix-registry-embedded-images/repair.py --help`
Expected: argparse help text prints, exit 0.

- [ ] **Step 5: Commit**

```bash
git add experiments/2026-07-01-fix-registry-embedded-images/
git commit -m "feat(experiments): add registry embedded-images audit + repair CLIs"
```

---

## Final Verification (after all tasks)

- [ ] **Run the full test suite in the sleap env:**

Run: `conda run -n sleap_v1.4.1 --no-capture-output python -m pytest tests/test_datasets.py -v`
Expected: all pass.

- [ ] **Confirm lint/format is clean:**

Run: `python -m black --check sleap_roots_training/datasets.py tests/test_datasets.py tests/fixtures.py experiments/2026-07-01-fix-registry-embedded-images/`
Expected: "All done!" (no reformatting needed).

- [ ] **Run the real read-only audit and review with Elizabeth (rollout step, not automated):**

Run: `cd experiments/2026-07-01-fix-registry-embedded-images && conda run -n sleap_v1.4.1 --no-capture-output python audit.py --output audit_results.csv`
Expected: a table of the 8 collections with `embedded`/`recoverable_via`; review the CSV before any `--apply`.
```
