import pytest
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch, MagicMock

from sleap_roots_training.datasets import make_dataset_artifact
from tests.fixtures import embedded_package, nonembedded_package, _build_tiny_labels


class TestMakeDatasetArtifact:
    """Test suite for make_dataset_artifact function."""

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
