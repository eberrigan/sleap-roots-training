import pytest
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch, MagicMock

from sleap_roots_training.datasets import make_dataset_artifact
from tests.fixtures import embedded_package, nonembedded_package, _build_tiny_labels


class TestMakeDatasetArtifact:
    """Test suite for make_dataset_artifact function."""

    @pytest.fixture(autouse=True)
    def _bypass_embedding_guard(self):
        with patch(
            "sleap_roots_training.datasets.has_embedded_images", return_value=True
        ):
            yield

    @patch("sleap_roots_training.datasets.wandb.init")
    @patch("sleap_roots_training.datasets.CONFIG")
    def test_make_dataset_artifact_basic(self, mock_config, mock_wandb_init):
        """Test basic dataset artifact creation."""
        # Setup mocks
        mock_config.__getitem__.side_effect = lambda key: {
            "project_name": "test_project",
            "entity_name": "test_entity",
            "experiment_name": "test_experiment",
            "registry": "test_registry",
            "collection_name": "test_collection",
        }[key]

        mock_run = MagicMock()
        mock_wandb_init.return_value = mock_run
        mock_artifact = MagicMock()

        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a test dataset file
            dataset_path = Path(temp_dir) / "test_dataset.slp"
            dataset_path.write_text("test dataset content")

            with patch(
                "sleap_roots_training.datasets.wandb.Artifact",
                return_value=mock_artifact,
            ):
                result = make_dataset_artifact(
                    artifact_name="test_artifact",
                    dataset_path=str(dataset_path),
                    link_to_registry=False,
                    description="Test dataset",
                    tags=["test", "dataset"],
                )

                # Assertions
                mock_wandb_init.assert_called_once_with(
                    project="test_project",
                    entity="test_entity",
                    job_type="build_dataset",
                    name="test_experiment",
                    save_code=True,
                )

                mock_artifact.add_file.assert_called_once_with(
                    local_path=dataset_path.as_posix(), overwrite=False
                )
                mock_run.log_artifact.assert_called_once_with(
                    mock_artifact, tags=["test", "dataset"]
                )
                mock_run.finish.assert_called_once()
                assert result == mock_artifact

    @patch("sleap_roots_training.datasets.wandb.init")
    @patch("sleap_roots_training.datasets.CONFIG")
    def test_make_dataset_artifact_with_registry_link(
        self, mock_config, mock_wandb_init
    ):
        """Test dataset artifact creation with registry link."""
        # Setup mocks
        mock_config.__getitem__.side_effect = lambda key: {
            "project_name": "test_project",
            "entity_name": "test_entity",
            "experiment_name": "test_experiment",
            "registry": "test_registry",
            "collection_name": "test_collection",
        }[key]

        mock_run = MagicMock()
        mock_wandb_init.return_value = mock_run
        mock_artifact = MagicMock()

        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_path = Path(temp_dir) / "test_dataset.slp"
            dataset_path.write_text("test dataset content")

            with patch(
                "sleap_roots_training.datasets.wandb.Artifact",
                return_value=mock_artifact,
            ):
                result = make_dataset_artifact(
                    artifact_name="test_artifact",
                    dataset_path=str(dataset_path),
                    link_to_registry=True,
                    description="Test dataset",
                    tags=["test", "dataset"],
                )

                # Assertions
                mock_run.link_artifact.assert_called_once_with(
                    artifact=mock_artifact,
                    target_path="test_entity-org/wandb-registry-test_registry/test_collection",
                )
                assert result == mock_artifact

    @patch("sleap_roots_training.datasets.wandb.init")
    @patch("sleap_roots_training.datasets.CONFIG")
    def test_make_dataset_artifact_no_description_no_tags(
        self, mock_config, mock_wandb_init
    ):
        """Test dataset artifact creation without description and tags."""
        # Setup mocks
        mock_config.__getitem__.side_effect = lambda key: {
            "project_name": "test_project",
            "entity_name": "test_entity",
            "experiment_name": "test_experiment",
            "registry": "test_registry",
            "collection_name": "test_collection",
        }[key]

        mock_run = MagicMock()
        mock_wandb_init.return_value = mock_run
        mock_artifact = MagicMock()

        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_path = Path(temp_dir) / "test_dataset.slp"
            dataset_path.write_text("test dataset content")

            with patch(
                "sleap_roots_training.datasets.wandb.Artifact",
                return_value=mock_artifact,
            ) as mock_artifact_class:
                result = make_dataset_artifact(
                    artifact_name="test_artifact", dataset_path=str(dataset_path)
                )

                # Assertions
                mock_artifact_class.assert_called_once_with(
                    name="test_artifact", type="dataset", description=""
                )
                mock_run.log_artifact.assert_called_once_with(mock_artifact, tags=None)
                assert result == mock_artifact

    @patch("sleap_roots_training.datasets.wandb.init")
    @patch("sleap_roots_training.datasets.CONFIG")
    def test_make_dataset_artifact_metadata_setting(self, mock_config, mock_wandb_init):
        """Test that metadata is set correctly."""
        # Setup mocks
        mock_config.__getitem__.side_effect = lambda key: {
            "project_name": "test_project",
            "entity_name": "test_entity",
            "experiment_name": "test_experiment",
            "registry": "test_registry",
            "collection_name": "test_collection",
        }[key]

        mock_run = MagicMock()
        mock_wandb_init.return_value = mock_run
        mock_artifact = MagicMock()
        mock_artifact.metadata = {}

        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_path = Path(temp_dir) / "test_dataset.slp"
            dataset_path.write_text("test dataset content")

            with patch(
                "sleap_roots_training.datasets.wandb.Artifact",
                return_value=mock_artifact,
            ):
                make_dataset_artifact(
                    artifact_name="test_artifact",
                    dataset_path=str(dataset_path),
                    tags=["tag1", "tag2"],
                )

                # Check metadata was set
                assert mock_artifact.metadata["data_path"] == dataset_path.as_posix()
                assert mock_artifact.metadata["tag1"] is True
                assert mock_artifact.metadata["tag2"] is True

    @patch("sleap_roots_training.datasets.wandb.init")
    @patch("sleap_roots_training.datasets.CONFIG")
    def test_make_dataset_artifact_nonexistent_file(self, mock_config, mock_wandb_init):
        """Test behavior with nonexistent dataset file."""
        # Setup mocks
        mock_config.__getitem__.side_effect = lambda key: {
            "project_name": "test_project",
            "entity_name": "test_entity",
            "experiment_name": "test_experiment",
            "registry": "test_registry",
            "collection_name": "test_collection",
        }[key]

        mock_run = MagicMock()
        mock_wandb_init.return_value = mock_run
        mock_artifact = MagicMock()

        nonexistent_path = "/nonexistent/path/dataset.slp"

        with patch(
            "sleap_roots_training.datasets.wandb.Artifact", return_value=mock_artifact
        ):
            # Should still try to add file (wandb will handle the error)
            result = make_dataset_artifact(
                artifact_name="test_artifact", dataset_path=nonexistent_path
            )

            mock_artifact.add_file.assert_called_once_with(
                local_path=nonexistent_path, overwrite=False
            )
            assert result == mock_artifact

    @patch("sleap_roots_training.datasets.wandb.init")
    @patch("sleap_roots_training.datasets.CONFIG")
    @patch("sleap_roots_training.datasets.logging")
    def test_make_dataset_artifact_exception_handling(
        self, mock_logging, mock_config, mock_wandb_init
    ):
        """Test exception handling in make_dataset_artifact."""
        # Setup mocks
        mock_config.__getitem__.side_effect = lambda key: {
            "project_name": "test_project",
            "entity_name": "test_entity",
            "experiment_name": "test_experiment",
            "registry": "test_registry",
            "collection_name": "test_collection",
        }[key]

        mock_run = MagicMock()
        mock_wandb_init.return_value = mock_run

        # Make wandb.Artifact raise an exception
        test_exception = Exception("Test exception")

        with patch(
            "sleap_roots_training.datasets.wandb.Artifact", side_effect=test_exception
        ):
            with pytest.raises(Exception, match="Test exception"):
                make_dataset_artifact(
                    artifact_name="test_artifact", dataset_path="/some/path/dataset.slp"
                )

            # Should still call run.finish() even on exception
            mock_run.finish.assert_called_once()
            mock_logging.error.assert_called_once_with(
                f"Error creating dataset artifact: {test_exception}"
            )

    @patch("sleap_roots_training.datasets.wandb.init")
    @patch("sleap_roots_training.datasets.CONFIG")
    @patch("sleap_roots_training.datasets.logging")
    def test_make_dataset_artifact_logging_messages(
        self, mock_logging, mock_config, mock_wandb_init
    ):
        """Test that appropriate logging messages are called."""
        # Setup mocks
        mock_config.__getitem__.side_effect = lambda key: {
            "project_name": "test_project",
            "entity_name": "test_entity",
            "experiment_name": "test_experiment",
            "registry": "test_registry",
            "collection_name": "test_collection",
        }[key]

        mock_run = MagicMock()
        mock_wandb_init.return_value = mock_run
        mock_artifact = MagicMock()

        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_path = Path(temp_dir) / "test_dataset.slp"
            dataset_path.write_text("test dataset content")

            with patch(
                "sleap_roots_training.datasets.wandb.Artifact",
                return_value=mock_artifact,
            ):
                make_dataset_artifact(
                    artifact_name="test_artifact",
                    dataset_path=str(dataset_path),
                    link_to_registry=True,
                )

                # Check logging calls
                mock_logging.info.assert_any_call(
                    f"Dataset artifact created: test_artifact from {dataset_path.as_posix()}."
                )
                mock_logging.info.assert_any_call(
                    "Linking test_artifact to registry test_entity-org/wandb-registry-test_registry/test_collection."
                )
                mock_logging.info.assert_any_call("W&B run finished successfully.")

    @patch("sleap_roots_training.datasets.wandb.init")
    @patch("sleap_roots_training.datasets.CONFIG")
    def test_make_dataset_artifact_path_conversion(self, mock_config, mock_wandb_init):
        """Test that string paths are converted to Path objects."""
        # Setup mocks
        mock_config.__getitem__.side_effect = lambda key: {
            "project_name": "test_project",
            "entity_name": "test_entity",
            "experiment_name": "test_experiment",
            "registry": "test_registry",
            "collection_name": "test_collection",
        }[key]

        mock_run = MagicMock()
        mock_wandb_init.return_value = mock_run
        mock_artifact = MagicMock()
        mock_artifact.metadata = {}

        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_path_str = str(Path(temp_dir) / "test_dataset.slp")
            Path(dataset_path_str).write_text("test dataset content")

            with patch(
                "sleap_roots_training.datasets.wandb.Artifact",
                return_value=mock_artifact,
            ):
                make_dataset_artifact(
                    artifact_name="test_artifact", dataset_path=dataset_path_str
                )

                # Check that the path was converted to posix format
                assert (
                    mock_artifact.metadata["data_path"]
                    == Path(dataset_path_str).as_posix()
                )
                mock_artifact.add_file.assert_called_once_with(
                    local_path=Path(dataset_path_str).as_posix(), overwrite=False
                )


class TestHasEmbeddedImages:
    """Tests for has_embedded_images detection."""

    def test_embedded_package_returns_true(self, embedded_package):
        from sleap_roots_training.datasets import has_embedded_images

        assert has_embedded_images(embedded_package) is True

    def test_nonembedded_package_returns_false(self, nonembedded_package):
        from sleap_roots_training.datasets import has_embedded_images

        assert has_embedded_images(nonembedded_package) is False

    def test_mixed_videos_ignores_non_user_videos(self):
        from sleap_roots_training.datasets import has_embedded_images

        class HDF5Video:  # class NAME must match what _video_has_embedded checks
            has_embedded_images = True

        class OtherVideo:  # a non-embedded, non-HDF5 backend
            pass

        video_a = SimpleNamespace(backend=HDF5Video())  # embedded, has user frames
        video_b = SimpleNamespace(backend=OtherVideo())  # NOT embedded, no user frames
        lf = SimpleNamespace(video=video_a, has_user_instances=True)
        fake_labels = SimpleNamespace(labeled_frames=[lf], videos=[video_a, video_b])
        fake_sleap = SimpleNamespace(load_file=lambda *a, **k: fake_labels)
        with patch("sleap_roots_training.datasets.sleap", fake_sleap):
            # video_b lacks embedded images but has no user frames -> ignored -> True
            assert has_embedded_images("whatever.slp") is True

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

    def test_search_paths_recovers_moved_image_sequence(self, tmp_path):
        pytest.importorskip("sleap")
        pytest.importorskip("imageio")
        from sleap_roots_training.datasets import inspect_package

        img_dir = tmp_path / "imgs"
        labels, _ = _build_tiny_labels(str(img_dir))
        out = str(tmp_path / "nonembedded.slp")
        labels.save(out, with_images=False)

        moved = tmp_path / "imgs_moved"
        img_dir.rename(moved)

        # Referenced images gone from their original location -> unrecoverable...
        assert inspect_package(out)["recoverable_via"] == "none"
        # ...but findable via search_paths (basename match) -> recoverable by re-embed.
        assert (
            inspect_package(out, search_paths=[str(moved)])["recoverable_via"]
            == "referenced_videos"
        )


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
            make_dataset_artifact(artifact_name="a", dataset_path="/tmp/broken.slp")
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


class TestRecoverabilityHelpers:
    """Tests for _classify_recoverability, _latest_version, _find_slp."""

    def test_classify_already_ok(self):
        from sleap_roots_training.datasets import _classify_recoverability

        assert (
            _classify_recoverability(
                {"embedded": True, "recoverable_via": "already_ok"}, False
            )
            == "already_ok"
        )

    def test_classify_already_embedded_beats_referenced(self):
        from sleap_roots_training.datasets import _classify_recoverability

        info = {"embedded": False, "recoverable_via": "referenced_videos"}
        assert (
            _classify_recoverability(info, data_path_embedded=True)
            == "already_embedded"
        )

    def test_classify_referenced_videos(self):
        from sleap_roots_training.datasets import _classify_recoverability

        info = {"embedded": False, "recoverable_via": "referenced_videos"}
        assert (
            _classify_recoverability(info, data_path_embedded=False)
            == "referenced_videos"
        )

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
