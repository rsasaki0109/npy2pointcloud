"""CLI interface for npy2pointcloud."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import click

from . import __version__
from .converter import SUPPORTED_FORMATS


@click.group()
@click.version_option(__version__)
@click.option("-v", "--verbose", is_flag=True, help="Enable verbose logging.")
def main(verbose: bool) -> None:
    """npy2pointcloud - Convert Rohbau3D .npy point clouds to standard formats."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(levelname)s: %(message)s",
        stream=sys.stderr,
    )


@main.command()
@click.option(
    "-i", "--input-dir",
    required=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Rohbau3D scene directory containing coord.npy, color.npy, etc.",
)
@click.option(
    "-o", "--output",
    required=True,
    type=click.Path(path_type=Path),
    help="Output file path (suffix will be set automatically).",
)
@click.option(
    "-f", "--format",
    "fmt",
    required=True,
    type=click.Choice(SUPPORTED_FORMATS, case_sensitive=False),
    help="Output format.",
)
def convert(input_dir: Path, output: Path, fmt: str) -> None:
    """Convert a single Rohbau3D scene to a point cloud file."""
    import time

    from .converter import convert as do_convert
    from .loader import load_scene

    click.echo(f"Loading scene from {input_dir} ...")
    t0 = time.monotonic()
    data = load_scene(input_dir)
    t_load = time.monotonic() - t0
    click.echo(f"  {data.num_points:,} points loaded ({t_load:.1f}s)")

    click.echo(f"Converting to {fmt.upper()} ...")
    t0 = time.monotonic()
    result = do_convert(data, output, fmt)
    t_conv = time.monotonic() - t0
    size_mb = result.stat().st_size / 1024 / 1024
    click.echo(f"  Written to {result} ({size_mb:.1f} MB, {t_conv:.1f}s)")


@main.command()
@click.option(
    "-i", "--input-dir",
    required=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Rohbau3D scene directory containing coord.npy.",
)
def info(input_dir: Path) -> None:
    """Show point cloud statistics for a Rohbau3D scene directory."""
    from .loader import load_scene

    data = load_scene(input_dir)
    click.echo(data.summary())


@main.command()
@click.option(
    "-i", "--input-dir",
    required=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Root directory of the Rohbau3D dataset.",
)
@click.option(
    "-o", "--output-dir",
    required=True,
    type=click.Path(path_type=Path),
    help="Root directory for output files.",
)
@click.option(
    "-f", "--format",
    "fmt",
    default="ply",
    type=click.Choice(SUPPORTED_FORMATS, case_sensitive=False),
    help="Output format (default: ply).",
)
@click.option(
    "--flatten",
    is_flag=True,
    help="Write all outputs into a flat directory instead of mirroring the tree.",
)
def batch(input_dir: Path, output_dir: Path, fmt: str, flatten: bool) -> None:
    """Batch-convert all scenes under a Rohbau3D dataset directory."""
    from .batch import batch_convert

    results = batch_convert(input_dir, output_dir, fmt, flatten=flatten)
    click.echo(f"\nDone. {len(results)} file(s) written.")
