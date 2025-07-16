# Status Badges for README

Add these badges to your README.md to show CI status:

```markdown
# sleap-roots-training

[![Branch Checks](https://github.com/eberrigan/sleap-roots-training/workflows/Branch%20Checks/badge.svg)](https://github.com/eberrigan/sleap-roots-training/actions/workflows/branch-checks.yml)
[![Test Imports](https://github.com/eberrigan/sleap-roots-training/workflows/Test%20Imports/badge.svg)](https://github.com/eberrigan/sleap-roots-training/actions/workflows/test-imports.yml)
[![CI with Conda](https://github.com/eberrigan/sleap-roots-training/workflows/CI%20with%20Conda/badge.svg)](https://github.com/eberrigan/sleap-roots-training/actions/workflows/ci-conda.yml)
[![CI](https://github.com/eberrigan/sleap-roots-training/workflows/CI/badge.svg)](https://github.com/eberrigan/sleap-roots-training/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/eberrigan/sleap-roots-training/branch/main/graph/badge.svg)](https://codecov.io/gh/eberrigan/sleap-roots-training)
```

## Badge Meanings

- **Branch Checks**: Essential fast checks (formatting, imports, basic tests)
- **Test Imports**: Cross-platform import compatibility
- **CI with Conda**: SLEAP integration testing with conda
- **CI**: Full integration testing with pip
- **codecov**: Code coverage reporting

## Workflow Triggers

All workflows run on:
- **Push to any branch** - immediate feedback during development
- **Pull requests to any branch** - validation before merging
- **Scheduled runs** (imports only) - daily monitoring

## Development Workflow

1. **Create feature branch** - `git checkout -b feature/my-feature`
2. **Make changes** - edit code, add tests
3. **Check locally** - `make test`, `make lint`
4. **Push branch** - triggers all CI workflows
5. **Monitor CI status** - ensure all checks pass
6. **Create pull request** - additional validation runs
7. **Merge when green** - all workflows must pass

## Troubleshooting CI Failures

### Branch Checks Failing
- **Code formatting**: Run `make format` locally
- **Import errors**: Check dependencies in pyproject.toml
- **Basic tests**: Run `pytest tests/test_imports.py tests/test_config.py`
- **SLEAP issues**: Verify SLEAP installation with `python verify_installation.py`

### Test Imports Failing
- **Cross-platform issues**: Check path handling, file operations
- **Python version compatibility**: Test with different Python versions locally
- **SLEAP PyPI installation**: Ensure `sleap[pypi]==1.4.1` installs correctly

### CI with Conda Failing
- **SLEAP conda installation**: Check platform-specific conda channels
- **Windows/Linux**: Uses `-c conda-forge -c nvidia -c sleap/label/dev -c sleap -c anaconda`
- **macOS**: Uses `-c conda-forge -c anaconda -c sleap`
- **Package conflicts**: Update dependency versions

### CI Failing
- **Full test suite**: Run `make test` locally
- **Coverage issues**: Add tests to increase coverage
- **Build problems**: Check pyproject.toml configuration
- **SLEAP PyPI issues**: Verify `sleap[pypi]==1.4.1` compatibility

## Installation Verification

Run the verification script to check your installation:

```bash
python verify_installation.py
```

This will check:
- Python version compatibility
- SLEAP installation
- Package installation
- Dependencies
- Configuration system