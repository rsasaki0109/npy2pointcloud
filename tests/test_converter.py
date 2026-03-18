"""Tests for npy2pointcloud.converter module."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import open3d as o3d
import pytest

from npy2pointcloud.converter import (
    SUPPORTED_FORMATS,
    _to_open3d,
    convert,
    to_las,
    to_pcd,
    to_ply,
)
from npy2pointcloud.loader import PointCloudData, load_scene


class TestToOpen3d:
    """Tests for _to_open3d helper."""

    def test_basic_conversion(self, scene_dir_coords_only: Path) -> None:
        data = load_scene(scene_dir_coords_only)
        pcd = _to_open3d(data)
        assert len(pcd.points) == data.num_points

    def test_colors_normalized(self, scene_dir_full: Path) -> None:
        data = load_scene(scene_dir_full)
        pcd = _to_open3d(data)
        colors = np.asarray(pcd.colors)
        assert colors.max() <= 1.0
        assert colors.min() >= 0.0

    def test_normals_preserved(self, scene_dir_full: Path) -> None:
        data = load_scene(scene_dir_full)
        pcd = _to_open3d(data)
        assert pcd.has_normals()


class TestToPly:
    """Tests for PLY conversion with round-trip verification."""

    def test_roundtrip_coords(self, scene_dir_coords_only: Path, tmp_path: Path) -> None:
        data = load_scene(scene_dir_coords_only)
        out = to_ply(data, tmp_path / "test.ply")
        assert out.exists()
        assert out.suffix == ".ply"

        loaded = o3d.io.read_point_cloud(str(out))
        coords_back = np.asarray(loaded.points)
        np.testing.assert_allclose(coords_back, data.coords, atol=1e-6)

    def test_roundtrip_with_colors(self, scene_dir_full: Path, tmp_path: Path) -> None:
        data = load_scene(scene_dir_full)
        out = to_ply(data, tmp_path / "full.ply")
        loaded = o3d.io.read_point_cloud(str(out))
        assert loaded.has_colors()
        assert len(loaded.points) == data.num_points

    def test_creates_parent_dirs(self, scene_dir_coords_only: Path, tmp_path: Path) -> None:
        data = load_scene(scene_dir_coords_only)
        out = to_ply(data, tmp_path / "sub" / "dir" / "test.ply")
        assert out.exists()


class TestToPcd:
    """Tests for PCD conversion with round-trip verification."""

    def test_roundtrip_coords(self, scene_dir_coords_only: Path, tmp_path: Path) -> None:
        data = load_scene(scene_dir_coords_only)
        out = to_pcd(data, tmp_path / "test.pcd")
        assert out.exists()
        assert out.suffix == ".pcd"

        loaded = o3d.io.read_point_cloud(str(out))
        coords_back = np.asarray(loaded.points)
        np.testing.assert_allclose(coords_back, data.coords, atol=1e-6)

    def test_roundtrip_with_colors_and_normals(self, scene_dir_full: Path, tmp_path: Path) -> None:
        data = load_scene(scene_dir_full)
        out = to_pcd(data, tmp_path / "full.pcd")
        loaded = o3d.io.read_point_cloud(str(out))
        assert loaded.has_colors()
        assert loaded.has_normals()
        assert len(loaded.points) == data.num_points


class TestToLas:
    """Tests for LAS conversion with round-trip verification."""

    def test_roundtrip_coords(self, scene_dir_coords_only: Path, tmp_path: Path) -> None:
        import laspy

        data = load_scene(scene_dir_coords_only)
        out = to_las(data, tmp_path / "test.las")
        assert out.exists()
        assert out.suffix == ".las"

        las = laspy.read(str(out))
        coords_back = np.column_stack([las.x, las.y, las.z])
        np.testing.assert_allclose(coords_back, data.coords, atol=0.002)

    def test_roundtrip_with_colors(self, scene_dir_full: Path, tmp_path: Path) -> None:
        import laspy

        data = load_scene(scene_dir_full)
        out = to_las(data, tmp_path / "full.las")
        las = laspy.read(str(out))
        assert len(las.points) == data.num_points
        # LAS stores 16-bit colors
        assert las.red is not None
        assert las.green is not None
        assert las.blue is not None

    def test_roundtrip_with_intensity(self, scene_dir_full: Path, tmp_path: Path) -> None:
        import laspy

        data = load_scene(scene_dir_full)
        out = to_las(data, tmp_path / "intensity.las")
        las = laspy.read(str(out))
        assert las.intensity is not None
        assert len(las.intensity) == data.num_points

    def test_roundtrip_with_normals(self, scene_dir_full: Path, tmp_path: Path) -> None:
        import laspy

        data = load_scene(scene_dir_full)
        out = to_las(data, tmp_path / "normals.las")
        las = laspy.read(str(out))
        nx = np.array(las.nx)
        np.testing.assert_allclose(nx, data.normals[:, 0].astype(np.float32), atol=1e-5)

    def test_las_version(self, scene_dir_coords_only: Path, tmp_path: Path) -> None:
        import laspy

        data = load_scene(scene_dir_coords_only)
        out = to_las(data, tmp_path / "version.las")
        las = laspy.read(str(out))
        assert las.header.version.major == 1
        assert las.header.version.minor == 4


class TestConvertDispatch:
    """Tests for the convert() dispatch function."""

    def test_supported_formats(self) -> None:
        assert "ply" in SUPPORTED_FORMATS
        assert "pcd" in SUPPORTED_FORMATS
        assert "las" in SUPPORTED_FORMATS

    def test_dispatch_ply(self, scene_dir_coords_only: Path, tmp_path: Path) -> None:
        data = load_scene(scene_dir_coords_only)
        out = convert(data, tmp_path / "dispatch", "ply")
        assert out.suffix == ".ply"
        assert out.exists()

    def test_dispatch_case_insensitive(self, scene_dir_coords_only: Path, tmp_path: Path) -> None:
        data = load_scene(scene_dir_coords_only)
        out = convert(data, tmp_path / "dispatch", "PLY")
        assert out.suffix == ".ply"

    def test_unsupported_format_raises(self, scene_dir_coords_only: Path, tmp_path: Path) -> None:
        data = load_scene(scene_dir_coords_only)
        with pytest.raises(ValueError, match="Unsupported format"):
            convert(data, tmp_path / "dispatch", "xyz")
