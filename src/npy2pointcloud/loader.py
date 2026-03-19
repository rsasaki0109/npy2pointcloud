"""Load Rohbau3D .npy point cloud files from a scene directory."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class PointCloudData:
    """Structured container for Rohbau3D point cloud data."""

    coords: np.ndarray  # (N, 3) float64 - XYZ coordinates
    colors: np.ndarray | None = None  # (N, 3) uint8 or float - RGB
    intensity: np.ndarray | None = None  # (N, 1) or (N,) - laser reflectance
    normals: np.ndarray | None = None  # (N, 3) float - surface normals
    source_dir: Path = field(default_factory=lambda: Path("."))

    @property
    def num_points(self) -> int:
        return self.coords.shape[0]

    @property
    def has_colors(self) -> bool:
        return self.colors is not None

    @property
    def has_intensity(self) -> bool:
        return self.intensity is not None

    @property
    def has_normals(self) -> bool:
        return self.normals is not None

    def summary(self) -> str:
        """Return a human-readable summary of the point cloud."""
        lines = [
            f"Source:     {self.source_dir}",
            f"Points:    {self.num_points:,}",
            f"XYZ range: x=[{self.coords[:, 0].min():.3f}, {self.coords[:, 0].max():.3f}]"
            f"  y=[{self.coords[:, 1].min():.3f}, {self.coords[:, 1].max():.3f}]"
            f"  z=[{self.coords[:, 2].min():.3f}, {self.coords[:, 2].max():.3f}]",
            f"Colors:    {'yes' if self.has_colors else 'no'}",
            f"Intensity: {'yes' if self.has_intensity else 'no'}",
            f"Normals:   {'yes' if self.has_normals else 'no'}",
        ]
        return "\n".join(lines)


def load_scene(scene_dir: str | Path) -> PointCloudData:
    """Load a Rohbau3D scene directory containing .npy files.

    Expected files:
        coord.npy   (required) - (N, 3) XYZ coordinates
        color.npy   (optional) - (N, 3) RGB values
        intensity.npy (optional) - (N, 1) laser reflectance
        normal.npy  (optional) - (N, 3) surface normals

    Parameters
    ----------
    scene_dir : path-like
        Directory containing the .npy files.

    Returns
    -------
    PointCloudData
        Loaded point cloud with all available attributes.

    Raises
    ------
    FileNotFoundError
        If coord.npy is missing.
    ValueError
        If array shapes are inconsistent.
    """
    scene_dir = Path(scene_dir)
    coord_path = scene_dir / "coord.npy"

    if not coord_path.exists():
        raise FileNotFoundError(f"Required file coord.npy not found in {scene_dir}")

    coords = np.load(coord_path).astype(np.float64)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError(f"coord.npy must be (N, 3), got {coords.shape}")

    n_points = coords.shape[0]

    # Load optional color
    colors = None
    color_path = scene_dir / "color.npy"
    if color_path.exists():
        colors = np.load(color_path)
        if colors.ndim != 2 or colors.shape[1] != 3:
            raise ValueError(f"color.npy must be (N, 3), got {colors.shape}")
        if colors.shape[0] != n_points:
            raise ValueError(
                f"color.npy has {colors.shape[0]} points but coord.npy has {n_points}"
            )

    # Load optional intensity
    intensity = None
    intensity_path = scene_dir / "intensity.npy"
    if intensity_path.exists():
        intensity = np.load(intensity_path)
        # Flatten (N, 1) to (N,)
        if intensity.ndim == 2 and intensity.shape[1] == 1:
            intensity = intensity.ravel()
        if intensity.shape[0] != n_points:
            raise ValueError(
                f"intensity.npy has {intensity.shape[0]} points but coord.npy has {n_points}"
            )

    # Load optional normals
    normals = None
    normal_path = scene_dir / "normal.npy"
    if normal_path.exists():
        normals = np.load(normal_path).astype(np.float64)
        if normals.ndim != 2 or normals.shape[1] != 3:
            raise ValueError(f"normal.npy must be (N, 3), got {normals.shape}")
        if normals.shape[0] != n_points:
            raise ValueError(
                f"normal.npy has {normals.shape[0]} points but coord.npy has {n_points}"
            )

    return PointCloudData(
        coords=coords,
        colors=colors,
        intensity=intensity,
        normals=normals,
        source_dir=scene_dir,
    )


def find_scenes(dataset_dir: str | Path) -> list[Path]:
    """Find all Rohbau3D scene directories under a dataset root.

    A scene directory is any directory containing a coord.npy file.

    Parameters
    ----------
    dataset_dir : path-like
        Root directory of the dataset.

    Returns
    -------
    list of Path
        Sorted list of scene directories.
    """
    dataset_dir = Path(dataset_dir)
    scenes = sorted(
        p.parent for p in dataset_dir.rglob("coord.npy")
    )
    return scenes
