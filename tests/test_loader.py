"""Tests for npy2pointcloud.loader module."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from npy2pointcloud.loader import PointCloudData, find_scenes, load_scene


class TestLoadScene:
    """Tests for load_scene()."""

    def test_load_full_scene(self, scene_dir_full: Path) -> None:
        data = load_scene(scene_dir_full)
        assert data.num_points == 100
        assert data.coords.shape == (100, 3)
        assert data.has_colors
        assert data.has_intensity
        assert data.has_normals
        assert data.source_dir == scene_dir_full

    def test_load_coords_only(self, scene_dir_coords_only: Path) -> None:
        data = load_scene(scene_dir_coords_only)
        assert data.num_points == 50
        assert not data.has_colors
        assert not data.has_intensity
        assert not data.has_normals

    def test_load_with_colors(self, scene_dir_with_colors: Path) -> None:
        data = load_scene(scene_dir_with_colors)
        assert data.num_points == 80
        assert data.has_colors
        assert not data.has_intensity

    def test_missing_coord_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="coord.npy"):
            load_scene(tmp_path)

    def test_bad_coord_shape_raises(self, tmp_path: Path) -> None:
        np.save(tmp_path / "coord.npy", np.random.rand(10, 4))
        with pytest.raises(ValueError, match="must be.*N, 3"):
            load_scene(tmp_path)

    def test_mismatched_color_count_raises(self, tmp_path: Path) -> None:
        np.save(tmp_path / "coord.npy", np.random.rand(10, 3))
        np.save(tmp_path / "color.npy", np.random.rand(5, 3))
        with pytest.raises(ValueError, match="color.npy has 5 points"):
            load_scene(tmp_path)

    def test_mismatched_intensity_count_raises(self, tmp_path: Path) -> None:
        np.save(tmp_path / "coord.npy", np.random.rand(10, 3))
        np.save(tmp_path / "intensity.npy", np.random.rand(5, 1))
        with pytest.raises(ValueError, match="intensity.npy has 5 points"):
            load_scene(tmp_path)

    def test_mismatched_normal_count_raises(self, tmp_path: Path) -> None:
        np.save(tmp_path / "coord.npy", np.random.rand(10, 3))
        np.save(tmp_path / "normal.npy", np.random.rand(5, 3))
        with pytest.raises(ValueError, match="normal.npy has 5 points"):
            load_scene(tmp_path)

    def test_intensity_shape_flattened(self, tmp_path: Path) -> None:
        """intensity.npy with shape (N, 1) should be flattened to (N,)."""
        n = 20
        np.save(tmp_path / "coord.npy", np.random.rand(n, 3))
        np.save(tmp_path / "intensity.npy", np.random.rand(n, 1))
        data = load_scene(tmp_path)
        assert data.intensity.ndim == 1
        assert data.intensity.shape == (n,)


class TestFindScenes:
    """Tests for find_scenes()."""

    def test_find_single_scene(self, scene_dir_full: Path) -> None:
        scenes = find_scenes(scene_dir_full.parent)
        assert scene_dir_full in scenes

    def test_find_nested_scenes(self, tmp_path: Path) -> None:
        for name in ["scene_a", "scene_b"]:
            d = tmp_path / name
            d.mkdir()
            np.save(d / "coord.npy", np.random.rand(5, 3))
        scenes = find_scenes(tmp_path)
        assert len(scenes) == 2
        names = [s.name for s in scenes]
        assert "scene_a" in names
        assert "scene_b" in names

    def test_find_no_scenes(self, tmp_path: Path) -> None:
        scenes = find_scenes(tmp_path)
        assert scenes == []


class TestPointCloudData:
    """Tests for PointCloudData dataclass."""

    def test_summary_output(self, scene_dir_full: Path) -> None:
        data = load_scene(scene_dir_full)
        summary = data.summary()
        assert "Points:" in summary
        assert "100" in summary
        assert "Colors:    yes" in summary
        assert "Intensity: yes" in summary
        assert "Normals:   yes" in summary

    def test_summary_no_optionals(self, scene_dir_coords_only: Path) -> None:
        data = load_scene(scene_dir_coords_only)
        summary = data.summary()
        assert "Colors:    no" in summary
        assert "Intensity: no" in summary
        assert "Normals:   no" in summary
