# npy2pointcloud

[![CI](https://github.com/rsasaki0109/npy2pointcloud/actions/workflows/ci.yml/badge.svg)](https://github.com/rsasaki0109/npy2pointcloud/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Convert [Rohbau3D](https://huggingface.co/datasets/Finnish-NLP/Rohbau3D) `.npy` point cloud files to standard formats (PLY, PCD, LAS).

## Supported Formats

| Format | Extension | Library | Colors | Normals | Intensity | Notes |
|--------|-----------|---------|--------|---------|-----------|-------|
| PLY    | `.ply`    | Open3D  | Yes    | Yes     | --        | Binary format, widely supported |
| PCD    | `.pcd`    | Open3D  | Yes    | Yes     | --        | Point Cloud Library native format |
| LAS    | `.las`    | laspy   | Yes (16-bit) | Yes (extra dims) | Yes | LAS 1.4, point format 7 |

## Rohbau3D Input Format

Each scene directory contains:

| File | Shape | Description |
|------|-------|-------------|
| `coord.npy` | (N, 3) float | XYZ coordinates (required) |
| `color.npy` | (N, 3) uint8 | RGB values |
| `intensity.npy` | (N, 1) float | Laser reflectance |
| `normal.npy` | (N, 3) float | Surface normals |

## Installation

```bash
pip install -e .
```

## Usage

### Convert a single scene

```bash
npy2pointcloud convert -i /path/to/scene -o output.ply -f ply
npy2pointcloud convert -i /path/to/scene -o output.las -f las
npy2pointcloud convert -i /path/to/scene -o output.pcd -f pcd
```

### Show point cloud info

```bash
npy2pointcloud info -i /path/to/scene
```

Example output:

```
Source:     /data/rohbau3d/scene_001
Points:    10,452,301
XYZ range: x=[-12.345, 45.678]  y=[-8.901, 23.456]  z=[0.123, 15.789]
Colors:    yes
Intensity: yes
Normals:   yes
```

### Batch convert an entire dataset

```bash
# Mirror directory structure
npy2pointcloud batch -i /path/to/dataset -o /path/to/output -f ply

# Flatten into a single directory
npy2pointcloud batch -i /path/to/dataset -o /path/to/output -f las --flatten
```

## Python API

```python
from npy2pointcloud.loader import load_scene
from npy2pointcloud.converter import convert

data = load_scene("/path/to/scene")
print(data.summary())

convert(data, "output.ply", "ply")
convert(data, "output.las", "las")
convert(data, "output.pcd", "pcd")
```

## Development

```bash
pip install -e .
pip install pytest
pytest tests/ -v
```
