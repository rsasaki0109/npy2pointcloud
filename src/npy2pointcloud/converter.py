"""Convert PointCloudData to standard point cloud formats (PLY, PCD, LAS)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import open3d as o3d

from .loader import PointCloudData


def _to_open3d(data: PointCloudData) -> o3d.geometry.PointCloud:
    """Convert PointCloudData to an Open3D PointCloud object."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(data.coords)

    if data.has_colors:
        colors = data.colors.astype(np.float64)
        # Normalize to [0, 1] if values are in [0, 255]
        if colors.max() > 1.0:
            colors = colors / 255.0
        pcd.colors = o3d.utility.Vector3dVector(colors)

    if data.has_normals:
        pcd.normals = o3d.utility.Vector3dVector(data.normals)

    return pcd


def to_ply(data: PointCloudData, output_path: str | Path) -> Path:
    """Write point cloud to PLY format using Open3D.

    Preserves XYZ, RGB, and normals. Intensity is stored as a custom
    comment (PLY does not have a standard intensity field in Open3D's writer).

    Parameters
    ----------
    data : PointCloudData
        Input point cloud.
    output_path : path-like
        Destination .ply file path.

    Returns
    -------
    Path
        The written file path.
    """
    output_path = Path(output_path).with_suffix(".ply")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    pcd = _to_open3d(data)
    o3d.io.write_point_cloud(str(output_path), pcd, write_ascii=False)
    return output_path


def to_pcd(data: PointCloudData, output_path: str | Path) -> Path:
    """Write point cloud to PCD format using Open3D.

    Preserves XYZ, RGB, and normals.

    Parameters
    ----------
    data : PointCloudData
        Input point cloud.
    output_path : path-like
        Destination .pcd file path.

    Returns
    -------
    Path
        The written file path.
    """
    output_path = Path(output_path).with_suffix(".pcd")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    pcd = _to_open3d(data)
    o3d.io.write_point_cloud(str(output_path), pcd, write_ascii=False)
    return output_path


def to_las(data: PointCloudData, output_path: str | Path) -> Path:
    """Write point cloud to LAS 1.4 format using laspy.

    Preserves XYZ, RGB (as 16-bit), intensity, and classification.
    Normals are stored as extra dimensions (nx, ny, nz).

    Parameters
    ----------
    data : PointCloudData
        Input point cloud.
    output_path : path-like
        Destination .las file path.

    Returns
    -------
    Path
        The written file path.
    """
    import laspy

    output_path = Path(output_path).with_suffix(".las")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Use point format 2 (XYZ + RGB) or 3 (XYZ + RGB + GPS time) or 6+ for 1.4
    # Point format 7 supports RGB in LAS 1.4
    header = laspy.LasHeader(point_format=7, version="1.4")

    # Compute scale and offset for coordinates
    mins = data.coords.min(axis=0)
    maxs = data.coords.max(axis=0)
    ranges = maxs - mins

    # Use offset at the minimum values, scale to preserve ~1mm precision
    header.offsets = mins
    header.scales = np.array([0.001, 0.001, 0.001])

    # Add extra dims for normals if present
    if data.has_normals:
        header.add_extra_dim(laspy.ExtraBytesParams(name="nx", type=np.float32, description="Normal X"))
        header.add_extra_dim(laspy.ExtraBytesParams(name="ny", type=np.float32, description="Normal Y"))
        header.add_extra_dim(laspy.ExtraBytesParams(name="nz", type=np.float32, description="Normal Z"))

    las = laspy.LasData(header)

    # Set coordinates
    las.x = data.coords[:, 0]
    las.y = data.coords[:, 1]
    las.z = data.coords[:, 2]

    # Set RGB (LAS uses 16-bit color values)
    if data.has_colors:
        colors = data.colors.astype(np.float64)
        if colors.max() <= 1.0:
            # Float [0, 1] -> 16-bit [0, 65535]
            las.red = (colors[:, 0] * 65535).astype(np.uint16)
            las.green = (colors[:, 1] * 65535).astype(np.uint16)
            las.blue = (colors[:, 2] * 65535).astype(np.uint16)
        else:
            # Uint8 [0, 255] -> 16-bit [0, 65535]
            las.red = (colors[:, 0] * 257).astype(np.uint16)  # 255 * 257 = 65535
            las.green = (colors[:, 1] * 257).astype(np.uint16)
            las.blue = (colors[:, 2] * 257).astype(np.uint16)

    # Set intensity (LAS uses 16-bit unsigned)
    if data.has_intensity:
        intensity = data.intensity.astype(np.float64)
        # Normalize to [0, 65535] range
        i_min, i_max = intensity.min(), intensity.max()
        if i_max > i_min:
            normalized = ((intensity - i_min) / (i_max - i_min) * 65535).astype(np.uint16)
        else:
            normalized = np.zeros(len(intensity), dtype=np.uint16)
        las.intensity = normalized

    # Set normals as extra dimensions
    if data.has_normals:
        las.nx = data.normals[:, 0].astype(np.float32)
        las.ny = data.normals[:, 1].astype(np.float32)
        las.nz = data.normals[:, 2].astype(np.float32)

    las.write(str(output_path))
    return output_path


# Format dispatch table
CONVERTERS = {
    "ply": to_ply,
    "pcd": to_pcd,
    "las": to_las,
}

SUPPORTED_FORMATS = list(CONVERTERS.keys())


def convert(data: PointCloudData, output_path: str | Path, fmt: str) -> Path:
    """Convert point cloud data to the specified format.

    Parameters
    ----------
    data : PointCloudData
        Input point cloud.
    output_path : path-like
        Destination file path (suffix will be adjusted).
    fmt : str
        Output format: "ply", "pcd", or "las".

    Returns
    -------
    Path
        The written file path.

    Raises
    ------
    ValueError
        If format is not supported.
    """
    fmt = fmt.lower()
    if fmt not in CONVERTERS:
        raise ValueError(f"Unsupported format '{fmt}'. Choose from: {SUPPORTED_FORMATS}")
    return CONVERTERS[fmt](data, output_path)
