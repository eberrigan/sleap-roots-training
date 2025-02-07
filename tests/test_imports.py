import pytest

def test_package_import():
    """Check that sleap_roots_training is importable."""
    try:
        import sleap_roots_training
        assert hasattr(sleap_roots_training, "__file__"), "Package does not have a __file__ attribute"
    except ModuleNotFoundError:
        pytest.fail("Failed to import sleap_roots_training")

def test_excluded_directories():
    """Ensure helper_functions and tests are not part of the installed package."""
    with pytest.raises(ModuleNotFoundError):
        import helper_functions  # Should fail since it's excluded

    with pytest.raises(ModuleNotFoundError):
        import tests  # Should fail since it's excluded
