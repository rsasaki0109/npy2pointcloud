"""Shared pytest fixtures for npy2pointcloud tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


@pytest.fixture()
def scene_dir_full(tmp_path: Path) -> Path:
    """Create a temporary scene directory with all .npy files (coords, colors, intensity, normals)."""
    n = 100
    np.save(tmp_path / "coord.npy", np.random.rand(n, 3).astype(np.float64))
    np.save(tmp_path / "color.npy", np.random.randint(0, 256, (n, 3)).astype(np.uint8))
    np.save(tmp_path / "intensity.npy", np.random.rand(n, 1).astype(np.float64))
    np.save(tmp_path / "normal.npy", np.random.rand(n, 3).astype(np.float64))
    return tmp_path


@pytest.fixture()
def scene_dir_coords_only(tmp_path: Path) -> Path:
    """Create a temporary scene directory with only coord.npy."""
    n = 50
    np.save(tmp_path / "coord.npy", np.random.rand(n, 3).astype(np.float64))
    return tmp_path


@pytest.fixture()
def scene_dir_with_colors(tmp_path: Path) -> Path:
    """Create a temporary scene directory with coords and colors."""
    scene = tmp_path / "scene_rgb"
    scene.mkdir()
    n = 80
    np.save(scene / "coord.npy", np.random.rand(n, 3).astype(np.float64))
    np.save(scene / "color.npy", np.random.randint(0, 256, (n, 3)).astype(np.uint8))
    return scene
