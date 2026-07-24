#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
# skeleton_visu

A lightweight visualization module that converts point‑cloud and skeleton data into PyVista mesh objects, enabling fast 3‑D rendering with just a few lines of code.
It simplifies the workflow for researchers and developers who need to inspect or present geometric data without handling the low‑level mesh construction themselves.

## Key Features
- **Point‑cloud conversion**: `pointcloud_polydata` builds a `pv.PolyData` mesh directly from an `(N, 3)` NumPy array of XYZ coordinates.
- **Skeleton rendering**: `skeleton_polydata` creates a `pv.PolyData` mesh with line cells from a dictionary containing `"points"` and `"lines"` arrays.
- **Zero‑boilerplate usage**: Both helpers return ready‑to‑plot PyVista objects, so you can immediately add them to a `pv.Plotter`.
- **No external dependencies beyond PyVista and NumPy**: The functions rely only on these common scientific packages.

## Usage Example

```python
>>> import pyvista as pv
>>> from skeleton_refinement.data import pointcloud
>>> from skeleton_refinement.data import skeleton
>>> from skeleton_refinement.visu import pointcloud_polydata
>>> from skeleton_refinement.visu import skeleton_polydata
>>> from skeleton_refinement.stochastic_registration import perform_registration
>>> pcd = pointcloud()
>>> skel = skeleton()
>>> ref_skel_pts = perform_registration(pcd, skel["points"], alpha=5, beta=10)
>>> ref_skel = {"points": ref_skel_pts, "lines": skel["lines"]}
>>> # Create a PyVista rendering:
>>> pcd_mesh = pointcloud_polydata(pcd)
>>> skel_mesh = skeleton_polydata(skel)
>>> ref_skel_mesh = skeleton_polydata(ref_skel)
>>> plotter = pv.Plotter()
>>> plotter.add_mesh(pcd_mesh, color='darkgreen', opacity=0.3)
>>> plotter.add_mesh(skel_mesh, color='red', line_width=3)
>>> plotter.add_mesh(ref_skel_mesh, color='blue', line_width=3)
>>> plotter.show_grid()
>>> plotter.add_floor()
>>> plotter.show()
```
"""

import numpy as np
import pyvista as pv


def pointcloud_polydata(points) -> "pv.PolyData":
    """
    Build a `pyvista.PolyData` mesh from the bundled point‑cloud.

    Parameters
    ----------
    points : np.ndarray
        An ``(M, 3)`` numpy array, representing the XYZ coordinates of the point‑cloud.

    Returns
    -------
    pyvista.PolyData
        A mesh whose vertices directly correspond to the XYZ coordinates of the point‑cloud.

    Examples
    --------
    >>> import pyvista as pv
    >>> from skeleton_refinement.data import pointcloud
    >>> from skeleton_refinement.visu import pointcloud_polydata
    >>> pcd = pointcloud_polydata()
    >>> pcd.n_points
    57890
    >>> # Create a PyVista rendering:
    >>> plotter = pv.Plotter(pointcloud())
    >>> plotter.add_mesh(pcd, color='darkgreen', opacity=0.7)
    >>> plotter.show_grid()
    >>> plotter.add_floor()
    >>> plotter.show()
    """
    # PyVista's PolyData constructor accepts an (N, 3) array of points.
    return pv.PolyData(points)


def skeleton_polydata(skel) -> "pv.PolyData":
    """
    Build a `pyvista.PolyData` mesh that visualizes the skeleton as a set of line segments
    connecting the skeleton points.

    Parameters
    ----------
    skel : dict
        A dictionary with "points" and "lines", representing the skeleton.
        The "points" entry is an ``(M, 3)`` numpy array, representing the XYZ coordinates of the skeleton points.
        The "lines" entry is a ``(L, 2)`` numpy array, representing the line segments connecting the skeleton points.

    Returns
    -------
    pyvista.PolyData
        A mesh whose points are the skeleton vertices and whose line cells are
        taken from the ``lines`` array in the original JSON file.

    Examples
    --------
    >>> import pyvista as pv
    >>> from skeleton_refinement.data import skeleton
    >>> from skeleton_refinement.visu import skeleton_polydata
    >>> skel_mesh = skeleton_polydata(skeleton())
    >>> skel_mesh.n_cells
    947   # number of line segments (len(lines))
    >>> # Create a PyVista rendering:
    >>> plotter = pv.Plotter()
    >>> plotter.add_mesh(skel_mesh, color='red')
    >>> plotter.show_grid()
    >>> plotter.add_floor()
    >>> plotter.show()
    """
    points: np.ndarray = np.asarray(skel["points"])  # (M, 3)
    lines: np.ndarray = np.asarray(skel["lines"], dtype=np.int32)  # (L, 2)

    # PyVista expects a *connectivity array* where each line is encoded as:
    # [n_vertices, idx0, idx1, ...]   (n_vertices == 2 for simple segments)
    # Build that array in a vectorized fashion.
    n_segments = lines.shape[0]
    # Shape (L, 1) filled with 2 (two vertices per line)
    seg_len = np.full((n_segments, 1), 2, dtype=np.int32)
    # Concatenate to obtain (L, 3): [2, v0, v1]
    connectivity = np.hstack([seg_len, lines])

    # Flatten to 1‑D as required by PyVista
    cell_array = connectivity.ravel()

    mesh = pv.PolyData()
    mesh.points = points
    mesh.lines = cell_array
    return mesh
