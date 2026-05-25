"""Tests for package setup — metadata, dependencies, and optional extras."""

from __future__ import annotations

from pathlib import Path

import tomllib

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

HERE = Path(__file__).resolve().parent.parent
PYPROJECT = HERE / "pyproject.toml"


def _load_pyproject() -> dict:
    """Parse pyproject.toml and return the root dict."""
    with open(PYPROJECT, "rb") as f:
        return tomllib.load(f)


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------


class TestMetadata:
    """Package metadata fields from pyproject.toml."""

    def test_package_name(self) -> None:
        data = _load_pyproject()
        assert data["project"]["name"] == "bipartite-gnn-gui"

    def test_package_version(self) -> None:
        data = _load_pyproject()
        assert data["project"]["version"] == "0.1.0"

    def test_package_description(self) -> None:
        data = _load_pyproject()
        desc = data["project"]["description"]
        assert "GNN" in desc
        assert "GUI" in desc

    def test_requires_python(self) -> None:
        data = _load_pyproject()
        assert data["project"]["requires-python"] == ">=3.10"

    def test_license(self) -> None:
        data = _load_pyproject()
        assert data["project"]["license"]["text"] == "MIT"


# ---------------------------------------------------------------------------
# Core dependencies
# ---------------------------------------------------------------------------


class TestCoreDependencies:
    """Core runtime dependencies declared in pyproject.toml."""

    def test_torch_dep(self) -> None:
        data = _load_pyproject()
        deps = data["project"]["dependencies"]
        assert any(d.startswith("torch>=") for d in deps), "torch not declared"

    def test_torch_geometric_dep(self) -> None:
        data = _load_pyproject()
        deps = data["project"]["dependencies"]
        assert any(
            d.startswith("torch-geometric>=") for d in deps
        ), "torch-geometric not declared"

    def test_numpy_dep(self) -> None:
        data = _load_pyproject()
        deps = data["project"]["dependencies"]
        assert any(d.startswith("numpy>=") for d in deps), "numpy not declared"

    def test_pillow_dep(self) -> None:
        data = _load_pyproject()
        deps = data["project"]["dependencies"]
        assert any(d.startswith("pillow>=") for d in deps), "pillow not declared"

    def test_pyyaml_dep(self) -> None:
        data = _load_pyproject()
        deps = data["project"]["dependencies"]
        assert any(d.startswith("pyyaml>=") for d in deps), "pyyaml not declared"

    def test_scipy_dep(self) -> None:
        data = _load_pyproject()
        deps = data["project"]["dependencies"]
        assert any(d.startswith("scipy>=") for d in deps), "scipy not declared"

    def test_pydantic_dep(self) -> None:
        data = _load_pyproject()
        deps = data["project"]["dependencies"]
        assert any(d.startswith("pydantic>=") for d in deps), "pydantic not declared"

    def test_tqdm_dep(self) -> None:
        data = _load_pyproject()
        deps = data["project"]["dependencies"]
        assert any(d.startswith("tqdm>=") for d in deps), "tqdm not declared"


# ---------------------------------------------------------------------------
# Optional extras
# ---------------------------------------------------------------------------


class TestOptionalExtras:
    """Optional dependency extras declared in pyproject.toml."""

    def test_wandb_extra(self) -> None:
        data = _load_pyproject()
        extras = data["project"]["optional-dependencies"]
        assert "wandb" in extras

    def test_wandb_version(self) -> None:
        data = _load_pyproject()
        deps = data["project"]["optional-dependencies"]["wandb"]
        assert any("wandb>=" in d for d in deps)

    def test_tensorboard_extra(self) -> None:
        data = _load_pyproject()
        extras = data["project"]["optional-dependencies"]
        assert "tensorboard" in extras

    def test_tensorboard_version(self) -> None:
        data = _load_pyproject()
        deps = data["project"]["optional-dependencies"]["tensorboard"]
        assert any("tensorboard>=" in d for d in deps)

    def test_dev_extra(self) -> None:
        data = _load_pyproject()
        extras = data["project"]["optional-dependencies"]
        assert "dev" in extras

    def test_dev_contains_pytest_cov(self) -> None:
        data = _load_pyproject()
        deps = data["project"]["optional-dependencies"]["dev"]
        assert any(d.startswith("pytest-cov>=") for d in deps)

    def test_test_extra(self) -> None:
        data = _load_pyproject()
        extras = data["project"]["optional-dependencies"]
        assert "test" in extras

    def test_all_extra(self) -> None:
        data = _load_pyproject()
        extras = data["project"]["optional-dependencies"]
        assert "all" in extras

    def test_all_combines_other_extras(self) -> None:
        """All extras should reference wandb, tensorboard, dev, and test."""
        data = _load_pyproject()
        all_deps = " ".join(data["project"]["optional-dependencies"]["all"])
        for sub in ("wandb", "tensorboard", "dev", "test"):
            assert sub in all_deps, f"all extra missing reference to [{sub}]"


# ---------------------------------------------------------------------------
# Build system
# ---------------------------------------------------------------------------


class TestBuildSystem:
    """Build system configuration."""

    def test_build_backend(self) -> None:
        data = _load_pyproject()
        assert data["build-system"]["build-backend"] == "setuptools.build_meta"

    def test_requires_setuptools(self) -> None:
        data = _load_pyproject()
        assert any("setuptools" in r for r in data["build-system"]["requires"])
