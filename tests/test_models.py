import pytest
import wandb
from unittest.mock import patch, MagicMock, Mock
from sleap_roots_training.models import (
    validate_tags,
    fetch_model_artifact_from_experiment,
    fetch_model_artifact_and_link_to_registry,
    promote_model_in_registry,
)


class TestValidateTags:
    """Test suite for validate_tags function."""

    def test_validate_tags_valid_input(self):
        """Test validate_tags with valid input."""
        valid_tags = ["tag1", "tag-2", "tag_3", "tag with spaces"]
        # Should not raise any exception
        validate_tags(valid_tags)

    def test_validate_tags_invalid_type(self):
        """Test validate_tags with invalid type."""
        with pytest.raises(ValueError, match="Tags should be a list of strings"):
            validate_tags("not_a_list")

    def test_validate_tags_invalid_tag_type(self):
        """Test validate_tags with invalid tag type."""
        with pytest.raises(ValueError, match="Tag '123' is not a string"):
            validate_tags(["valid_tag", 123])

    def test_validate_tags_invalid_characters(self):
        """Test validate_tags with invalid characters."""
        with pytest.raises(ValueError, match="Invalid W&B tag"):
            validate_tags(["valid_tag", "invalid@tag"])

    def test_validate_tags_empty_list(self):
        """Test validate_tags with empty list."""
        # Should not raise any exception
        validate_tags([])

    def test_validate_tags_special_characters(self):
        """Test validate_tags with various special characters."""
        # Valid characters
        valid_tags = ["tag-with-hyphens", "tag_with_underscores", "tag with spaces"]
        validate_tags(valid_tags)

        # Invalid characters
        invalid_chars = ["@", "#", "$", "%", "^", "&", "*", "(", ")", "=", "+"]
        for char in invalid_chars:
            with pytest.raises(ValueError, match="Invalid W&B tag"):
                validate_tags([f"tag{char}invalid"])


class TestFetchModelArtifactFromExperiment:
    """Test suite for fetch_model_artifact_from_experiment function."""

    @patch("sleap_roots_training.models.wandb.init")
    @patch("builtins.print")
    def test_fetch_model_artifact_from_experiment_latest(
        self, mock_print, mock_wandb_init
    ):
        """Test fetching latest version of model artifact."""
        # Setup mocks
        mock_run = MagicMock()
        mock_artifact = MagicMock()
        mock_artifact.download.return_value = "/path/to/artifact"
        mock_run.use_artifact.return_value = mock_artifact
        mock_wandb_init.return_value = mock_run

        # Call function
        result = fetch_model_artifact_from_experiment(
            "test_project", "test_entity", "test_artifact"
        )

        # Assertions
        mock_wandb_init.assert_called_once_with(
            project="test_project", entity="test_entity", job_type="fetch_artifact"
        )
        mock_run.use_artifact.assert_called_once_with("test_artifact:latest")
        mock_artifact.download.assert_called_once()
        mock_run.finish.assert_called_once()
        assert result == mock_artifact

    @patch("sleap_roots_training.models.wandb.init")
    @patch("builtins.print")
    def test_fetch_model_artifact_from_experiment_specific_version(
        self, mock_print, mock_wandb_init
    ):
        """Test fetching specific version of model artifact."""
        # Setup mocks
        mock_run = MagicMock()
        mock_artifact = MagicMock()
        mock_artifact.download.return_value = "/path/to/artifact"
        mock_run.use_artifact.return_value = mock_artifact
        mock_wandb_init.return_value = mock_run

        # Call function
        result = fetch_model_artifact_from_experiment(
            "test_project", "test_entity", "test_artifact", "v1"
        )

        # Assertions
        mock_run.use_artifact.assert_called_once_with("test_artifact:v1")
        assert result == mock_artifact

    @patch("sleap_roots_training.models.wandb.init")
    @patch("builtins.print")
    def test_fetch_model_artifact_print_messages(self, mock_print, mock_wandb_init):
        """Test that appropriate print messages are called."""
        # Setup mocks
        mock_run = MagicMock()
        mock_artifact = MagicMock()
        mock_artifact.download.return_value = "/path/to/artifact"
        mock_run.use_artifact.return_value = mock_artifact
        mock_wandb_init.return_value = mock_run

        # Call function
        fetch_model_artifact_from_experiment(
            "test_project", "test_entity", "test_artifact"
        )

        # Check print calls
        expected_calls = [
            ("Fetching artifact 'test_artifact:latest' from project 'test_project'.",),
            ("Fetched artifact 'test_artifact:latest'.",),
            (
                "Fetched artifact 'test_artifact:latest' to directory '/path/to/artifact'.",
            ),
        ]
        mock_print.assert_has_calls(
            [pytest.mock.call(*args) for args in expected_calls]
        )


class TestFetchModelArtifactAndLinkToRegistry:
    """Test suite for fetch_model_artifact_and_link_to_registry function."""

    @patch("sleap_roots_training.models.wandb.init")
    @patch("builtins.print")
    def test_fetch_and_link_to_registry(self, mock_print, mock_wandb_init):
        """Test fetching artifact and linking to registry."""
        # Setup mocks
        mock_run = MagicMock()
        mock_artifact = MagicMock()
        mock_run.use_artifact.return_value = mock_artifact
        mock_wandb_init.return_value = mock_run

        # Call function
        fetch_model_artifact_and_link_to_registry(
            "test_project",
            "test_entity",
            "test_artifact",
            "test_registry",
            "test_collection",
        )

        # Assertions
        mock_wandb_init.assert_called_once_with(
            project="test_project", entity="test_entity", job_type="fetch_artifact"
        )
        mock_run.use_artifact.assert_called_once_with("test_artifact:latest")
        mock_run.link_artifact.assert_called_once_with(
            mock_artifact,
            "test_entity-org/wandb-registry-test_registry/test_collection",
        )
        mock_run.finish.assert_called_once()

    @patch("sleap_roots_training.models.wandb.init")
    @patch("builtins.print")
    def test_fetch_and_link_specific_version(self, mock_print, mock_wandb_init):
        """Test fetching specific version and linking to registry."""
        # Setup mocks
        mock_run = MagicMock()
        mock_artifact = MagicMock()
        mock_run.use_artifact.return_value = mock_artifact
        mock_wandb_init.return_value = mock_run

        # Call function
        fetch_model_artifact_and_link_to_registry(
            "test_project",
            "test_entity",
            "test_artifact",
            "test_registry",
            "test_collection",
            "v2",
        )

        # Assertions
        mock_run.use_artifact.assert_called_once_with("test_artifact:v2")

    @patch("sleap_roots_training.models.wandb.init")
    @patch("builtins.print")
    def test_fetch_and_link_print_messages(self, mock_print, mock_wandb_init):
        """Test that appropriate print messages are called."""
        # Setup mocks
        mock_run = MagicMock()
        mock_artifact = MagicMock()
        mock_run.use_artifact.return_value = mock_artifact
        mock_wandb_init.return_value = mock_run

        # Call function
        fetch_model_artifact_and_link_to_registry(
            "test_project",
            "test_entity",
            "test_artifact",
            "test_registry",
            "test_collection",
        )

        # Check print calls
        expected_calls = [
            ("Fetching artifact 'test_artifact:latest' from project 'test_project'.",),
            ("Fetched artifact 'test_artifact:latest'.",),
            (
                "Linking artifact 'test_artifact:latest' to registry 'test_entity-org/wandb-registry-test_registry/test_collection'.",
            ),
            (
                "Linked artifact 'test_artifact:latest' to registry 'test_entity-org/wandb-registry-test_registry/test_collection'.",
            ),
        ]
        mock_print.assert_has_calls(
            [pytest.mock.call(*args) for args in expected_calls]
        )


class TestPromoteModelInRegistry:
    """Test suite for promote_model_in_registry function."""

    @patch("sleap_roots_training.models.wandb.init")
    @patch("builtins.print")
    def test_promote_model_in_registry(self, mock_print, mock_wandb_init):
        """Test promoting model in registry."""
        # Setup mocks
        mock_run = MagicMock()
        mock_artifact = MagicMock()
        mock_artifact.aliases = []
        mock_run.use_artifact.return_value = mock_artifact
        mock_wandb_init.return_value = mock_run

        # Call function
        promote_model_in_registry(
            "test_project",
            "test_entity",
            "test_registry",
            "test_artifact",
            "production",
        )

        # Assertions
        mock_wandb_init.assert_called_once_with(
            project="test_project",
            entity="test_entity",
            job_type="promote_registry_artifact",
        )
        mock_run.use_artifact.assert_called_once_with(
            "test_registry/test_artifact:latest:latest"
        )
        assert "production" in mock_artifact.aliases
        mock_artifact.save.assert_called_once()
        mock_run.finish.assert_called_once()

    @patch("sleap_roots_training.models.wandb.init")
    @patch("builtins.print")
    def test_promote_model_specific_version(self, mock_print, mock_wandb_init):
        """Test promoting specific version of model in registry."""
        # Setup mocks
        mock_run = MagicMock()
        mock_artifact = MagicMock()
        mock_artifact.aliases = []
        mock_run.use_artifact.return_value = mock_artifact
        mock_wandb_init.return_value = mock_run

        # Call function
        promote_model_in_registry(
            "test_project",
            "test_entity",
            "test_registry",
            "test_artifact",
            "staging",
            "v3",
        )

        # Assertions
        mock_run.use_artifact.assert_called_once_with(
            "test_registry/test_artifact:v3:latest"
        )
        assert "staging" in mock_artifact.aliases

    @patch("sleap_roots_training.models.wandb.init")
    @patch("builtins.print")
    def test_promote_model_print_message(self, mock_print, mock_wandb_init):
        """Test that promote_model prints confirmation message."""
        # Setup mocks
        mock_run = MagicMock()
        mock_artifact = MagicMock()
        mock_artifact.aliases = []
        mock_run.use_artifact.return_value = mock_artifact
        mock_wandb_init.return_value = mock_run

        # Call function
        promote_model_in_registry(
            "test_project",
            "test_entity",
            "test_registry",
            "test_artifact",
            "production",
        )

        # Check print call
        mock_print.assert_called_once_with(
            "Promoted artifact 'test_artifact:latest' in registry 'test_registry' to stage 'production'."
        )
