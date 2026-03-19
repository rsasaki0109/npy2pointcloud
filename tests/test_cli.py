"""Tests for npy2pointcloud.cli module."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from click.testing import CliRunner

from npy2pointcloud.cli import main


@pytest.fixture()
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture()
def scene_with_data(tmp_path: Path) -> Path:
    """Create a scene directory with coords and colors for CLI tests."""
    scene = tmp_path / "scene"
    scene.mkdir()
    n = 30
    np.save(scene / "coord.npy", np.random.rand(n, 3).astype(np.float64))
    np.save(scene / "color.npy", np.random.randint(0, 256, (n, 3)).astype(np.uint8))
    return scene


class TestVersionOption:
    def test_version(self, runner: CliRunner) -> None:
        result = runner.invoke(main, ["--version"])
        assert result.exit_code == 0
        assert "0.1.0" in result.output


class TestConvertCommand:
    def test_convert_ply(self, runner: CliRunner, scene_with_data: Path, tmp_path: Path) -> None:
        out = tmp_path / "out.ply"
        result = runner.invoke(main, ["convert", "-i", str(scene_with_data), "-o", str(out), "-f", "ply"])
        assert result.exit_code == 0
        assert "Written to" in result.output
        assert out.exists()

    def test_convert_pcd(self, runner: CliRunner, scene_with_data: Path, tmp_path: Path) -> None:
        out = tmp_path / "out.pcd"
        result = runner.invoke(main, ["convert", "-i", str(scene_with_data), "-o", str(out), "-f", "pcd"])
        assert result.exit_code == 0
        assert out.exists()

    def test_convert_las(self, runner: CliRunner, scene_with_data: Path, tmp_path: Path) -> None:
        out = tmp_path / "out.las"
        result = runner.invoke(main, ["convert", "-i", str(scene_with_data), "-o", str(out), "-f", "las"])
        assert result.exit_code == 0
        assert out.exists()

    def test_convert_missing_input(self, runner: CliRunner, tmp_path: Path) -> None:
        result = runner.invoke(main, ["convert", "-i", str(tmp_path / "nonexistent"), "-o", "out.ply", "-f", "ply"])
        assert result.exit_code != 0

    def test_convert_invalid_format(self, runner: CliRunner, scene_with_data: Path, tmp_path: Path) -> None:
        result = runner.invoke(main, ["convert", "-i", str(scene_with_data), "-o", str(tmp_path / "out"), "-f", "xyz"])
        assert result.exit_code != 0


class TestInfoCommand:
    def test_info(self, runner: CliRunner, scene_with_data: Path) -> None:
        result = runner.invoke(main, ["info", "-i", str(scene_with_data)])
        assert result.exit_code == 0
        assert "Points:" in result.output
        assert "30" in result.output
        assert "Colors:" in result.output

    def test_info_missing_dir(self, runner: CliRunner, tmp_path: Path) -> None:
        result = runner.invoke(main, ["info", "-i", str(tmp_path / "nonexistent")])
        assert result.exit_code != 0


class TestBatchCommand:
    def test_batch_convert(self, runner: CliRunner, tmp_path: Path) -> None:
        # Create a small dataset with two scenes
        for name in ["s1", "s2"]:
            d = tmp_path / "dataset" / name
            d.mkdir(parents=True)
            np.save(d / "coord.npy", np.random.rand(10, 3))

        out_dir = tmp_path / "output"
        result = runner.invoke(main, [
            "batch",
            "-i", str(tmp_path / "dataset"),
            "-o", str(out_dir),
            "-f", "ply",
        ])
        assert result.exit_code == 0
        assert "2 file(s) written" in result.output

    def test_batch_flatten(self, runner: CliRunner, tmp_path: Path) -> None:
        d = tmp_path / "dataset" / "scene1"
        d.mkdir(parents=True)
        np.save(d / "coord.npy", np.random.rand(10, 3))

        out_dir = tmp_path / "output"
        result = runner.invoke(main, [
            "batch",
            "-i", str(tmp_path / "dataset"),
            "-o", str(out_dir),
            "-f", "ply",
            "--flatten",
        ])
        assert result.exit_code == 0
        assert "1 file(s) written" in result.output


class TestVerboseOption:
    def test_verbose_flag(self, runner: CliRunner, scene_with_data: Path) -> None:
        result = runner.invoke(main, ["-v", "info", "-i", str(scene_with_data)])
        assert result.exit_code == 0
