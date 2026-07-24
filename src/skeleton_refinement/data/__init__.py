#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
# Access to Plant Skeleton and Point Cloud Datasets

Provides a lightweight interface for loading the bundled point‑cloud and its corresponding skeleton.

## Key Features

- **Path helpers**: `pointcloud_path()` and `skeleton_path()` return absolute `Path` objects to the packaged data files.
- **Data loaders**: `pointcloud()` reads the PLY file into a `(N, 3)` NumPy array; `skeleton_points()` extracts the skeleton point coordinates from the JSON file.
- **Transparent resource handling**: Uses `importlib.resources` to locate package data regardless of installation method (editable install, wheel, etc.).
- **Zero‑dependency API** for the end user: the only required external packages are NumPy and the internal `io` utilities.

## Usage Examples

```python
>>> from skeleton_refinement.data import pointcloud_path
>>> from skeleton_refinement.data import skeleton_path
>>> from skeleton_refinement.data import pointcloud
>>> from skeleton_refinement.data import skeleton_points
>>> # Get raw file locations
>>> print(pointcloud_path())
/.../skeleton_refinement/data/real_plant/PointCloud.ply
>>> print(skeleton_path())
/.../skeleton_refinement/data/real_plant/CurveSkeleton.json
>>> # Load data as NumPy arrays
>>> pcd = pointcloud()
>>> print(pcd.shape)
(57890, 3)
>>> skel = skeleton_points()
>>> print(skel.shape)
(948, 3)
```
"""

from importlib import resources
from pathlib import Path

import numpy as np

from ..io import load_json
from ..io import load_ply


def pointcloud_path() -> Path:
    """
    Get the absolute path to the bundled point‑cloud PLY file.

    Returns
    -------
    pathlib.Path
        Absolute file system path to ``PointCloud.ply`` located in the package resources.

    Examples
    --------
    >>> from skeleton_refinement.data import pointcloud_path
    >>> pcd_path = pointcloud_path()
    >>> print(pcd_path)  # doctest: +ELLIPSIS
    /.../skeleton_refinement/data/real_plant/PointCloud.ply
    """
    # Get the directory path to the package data files
    data_dir = resources.files('skeleton_refinement.data.real_plant')
    return Path(data_dir.joinpath("PointCloud.ply"))


def skeleton_path() -> Path:
    """
    Get the absolute path to the bundled skeleton JSON file.

    Returns
    -------
    pathlib.Path
        Absolute file system path to ``CurveSkeleton.json`` located in the package resources.

    Examples
    --------
    >>> from skeleton_refinement.data import skeleton_path
    >>> skel_path = skeleton_path()
    >>> print(skel_path)  # doctest: +ELLIPSIS
    /.../skeleton_refinement/data/real_plant/CurveSkeleton.json
    """
    # Get the directory path to the package data files
    data_dir = resources.files('skeleton_refinement.data.real_plant')
    return Path(data_dir.joinpath("CurveSkeleton.json"))


def pointcloud() -> np.ndarray:
    """
    Load the bundled point‑cloud data as a NumPy array.

    Returns
    -------
    numpy.ndarray
        Array of shape ``(N, 3)`` containing the XYZ coordinates of the point cloud.

    Examples
    --------
    >>> from skeleton_refinement.data import pointcloud
    >>> pcd = pointcloud()
    >>> print(pcd.shape)
    (57890, 3)
    """
    return load_ply(pointcloud_path())


def skeleton_points() -> np.ndarray:
    """
    Load the skeleton point coordinates from the bundled JSON file.

    Returns
    -------
    numpy.ndarray
        Array of shape ``(M, 3)`` containing the XYZ coordinates of the skeleton points.

    Examples
    --------
    >>> from skeleton_refinement.data import skeleton_points
    >>> skel = skeleton_points()
    >>> print(skel.shape)
    (948, 3)
    """
    return load_json(skeleton_path(), key="points")


def skeleton() -> dict[str, np.ndarray]:
    """
    Load the skeleton point coordinates and lines from the bundled JSON file.

    Returns
    -------
    dict
        The whole skeleton dictionary with "points" and "lines".

        - "points" with an array of shape ``(M, 3)`` containing the XYZ coordinates of the skeleton points.
        - "lines" with an array of shape ``(M, 2)`` containing the pais of point indexes joining them.

    Examples
    --------
    >>> from skeleton_refinement.data import skeleton
    >>> skel = skeleton()
    >>> print(list(skel.keys()))
    ['points', 'lines']
    """
    skel_data = load_json(skeleton_path())
    return {"points": np.asarray(skel_data["points"]), "lines": np.asarray(skel_data["lines"])}
