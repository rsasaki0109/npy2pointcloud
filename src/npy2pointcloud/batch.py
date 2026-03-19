"""Batch conversion of Rohbau3D dataset directories."""

from __future__ import annotations

import logging
from pathlib import Path

from .converter import convert, SUPPORTED_FORMATS
from .loader import find_scenes, load_scene

logger = logging.getLogger(__name__)


def batch_convert(
    dataset_dir: str | Path,
    output_dir: str | Path,
    fmt: str = "ply",
    *,
    flatten: bool = False,
) -> list[Path]:
    """Convert all scenes in a Rohbau3D dataset directory tree.

    Discovers all subdirectories containing coord.npy and converts each
    to the specified format.

    Parameters
    ----------
    dataset_dir : path-like
        Root directory of the Rohbau3D dataset.
    output_dir : path-like
        Root directory for output files.
    fmt : str
        Output format: "ply", "pcd", or "las".
    flatten : bool
        If True, write all outputs directly into output_dir with scene
        path encoded in the filename (e.g., "scene1__room2.ply").
        If False, mirror the directory structure under output_dir.

    Returns
    -------
    list of Path
        Paths to all written output files.
    """
    dataset_dir = Path(dataset_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    scenes = find_scenes(dataset_dir)
    if not scenes:
        logger.warning("No scenes found under %s", dataset_dir)
        return []

    logger.info("Found %d scene(s) under %s", len(scenes), dataset_dir)
    written: list[Path] = []

    for scene_dir in scenes:
        # Determine output path
        relative = scene_dir.relative_to(dataset_dir)
        if flatten:
            # Encode path separators as double underscores
            stem = str(relative).replace("/", "__").replace("\\", "__")
            out_path = output_dir / stem
        else:
            out_path = output_dir / relative / "pointcloud"

        try:
            logger.info("Loading %s ...", scene_dir)
            data = load_scene(scene_dir)
            logger.info(
                "  %s points, colors=%s, intensity=%s, normals=%s",
                f"{data.num_points:,}",
                data.has_colors,
                data.has_intensity,
                data.has_normals,
            )

            result_path = convert(data, out_path, fmt)
            written.append(result_path)
            logger.info("  -> %s", result_path)

        except Exception:
            logger.exception("Failed to convert %s", scene_dir)

    logger.info("Converted %d / %d scenes", len(written), len(scenes))
    return written
