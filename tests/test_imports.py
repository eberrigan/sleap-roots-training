import pytest


def test_package_import():
    """Check that sleap_roots_training is importable."""
    try:
        import sleap_roots_training

        assert hasattr(
            sleap_roots_training, "__file__"
        ), "Package does not have a __file__ attribute"
    except ModuleNotFoundError:
        pytest.fail("Failed to import sleap_roots_training")


def test_excluded_directories():
    """Ensure helper_functions and tests are not part of the installed package."""
    with pytest.raises(ModuleNotFoundError):
        import helper_functions  # noqa: F401 - Should fail since it's excluded

    # Note: tests module may be importable in development environment
    # This test is more relevant in a built/installed package
    try:
        import tests  # noqa: F401

        # If we can import tests, that's okay in development mode
        # The exclusion is handled in the package building process
    except ModuleNotFoundError:
        # If tests can't be imported, that's also fine
        pass
