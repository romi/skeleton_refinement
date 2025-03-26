#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
## 3D Skeleton and Point Cloud I/O

This module provides functions for loading and saving 3D point clouds and skeleton data in various file formats, simplifying data interchange between different tools and libraries.

### Key Features

- Load point clouds from XYZ, PLY and JSON formats
- Load skeleton data from NetworkX graph files
- Save tree structures to JSON and NetworkX pickle formats
- Support for various coordinate formats and attribute handling
"""

from pathlib import Path

import numpy as np


def load_xyz(filename):
    """Load a point cloud or skeleton file saved as a series of space-separated XYZ coordinates.

    Parameters
    ----------
    filename : str or pathlib.Path
        Path to the point cloud or skeleton file to parse.

    Returns
    -------
    numpy.ndarray
        The XYZ coordinates of the point cloud or skeleton as a NumPy array
        with shape ``(n, 3)``, where ``n`` is the number of points.

    Examples
    --------
    >>> from skeleton_refinement.io import load_xyz
    >>> points = load_xyz('point_cloud.xyz')
    >>> print(points.shape)
    (1000, 3)
    >>> print(points[:2])
    [[ 1.2  3.4  5.6]
     [-0.1  0.2  0.3]]
    """
    f = open(filename, "r")
    lines = f.readlines()
    org_x = []
    org_y = []
    org_z = []
    for l in lines:
        org_x.append(float(l.split(' ')[0]))
        org_y.append(float(l.split(' ')[1]))
        org_z.append(float(l.split(' ')[2]))
    f.close()
    X = np.column_stack((org_x, org_y, org_z))
    return X


def load_ply(filename):
    """Load point cloud coordinates from a PLY file.

    Parameters
    ----------
    filename : str or pathlib.Path
        Path to the PLY file to parse.

    Returns
    -------
    The XYZ coordinates of the point cloud as a NumPy array
        with shape ``(n, 3)``, where ``n`` is the number of points.

    Examples
    --------
    >>> from skeleton_refinement.io import load_ply
    >>> points = load_ply('point_cloud.ply')
    >>> print(points.shape)
    (1000, 3)
    >>> print(points[:2])
    [[ 1.2  3.4  5.6]
     [-0.1  0.2  0.3]]
    """
    from plyfile import PlyData
    plydata = PlyData.read(filename)
    X = np.array([plydata['vertex']['x'], plydata['vertex']['y'], plydata['vertex']['z']]).T
    return X


def load_json(filename, key=None):
    """Load a point cloud or skeleton file from a JSON file.

    Parameters
    ----------
    filename : str or pathlib.Path
        Path to the JSON file to parse.
    key : str, optional
        The key of the JSON dictionary containing the point cloud or skeleton
        coordinates to load. If ``None``, the entire JSON content is returned.

    Returns
    -------
    numpy.ndarray
        The XYZ coordinates of the point cloud or skeleton as a NumPy array
        with shape ``(n, 3)``, where ``n`` is the number of points.

    Examples
    --------
    >>> from skeleton_refinement.io import load_json
    >>> # JSON file with structure: {"points": [[x1,y1,z1], [x2,y2,z2], ...]}
    >>> points = load_json('point_cloud.json', key='points')
    >>> print(points.shape)
    (1000, 3)
    >>> # JSON file with direct array: [[x1,y1,z1], [x2,y2,z2], ...]
    >>> points = load_json('simple_points.json')
    >>> print(points.shape)
    (1000, 3)
    """
    import json
    with open(filename, mode='rb') as f:
        X = json.load(f)

    if key is not None:
        X = X[key]
    return np.array(X)


def load_nx(filename, key='position'):
    """Load a tree graph from a pickled NetworkX file.

    Parameters
    ----------
    filename : str or pathlib.Path
        Path to the pickled NetworkX graph file to parse.
    key : str, optional
        The node attribute key containing the position data.
        Default is 'position'.

    Returns
    -------
    numpy.ndarray
        The XYZ coordinates of the nodes as a NumPy array with shape (n, 3),
        where n is the number of nodes.

    Examples
    --------
    >>> from skeleton_refinement.io import load_nx
    >>> positions = load_nx('graph.pkl')
    >>> print(positions.shape)
    (50, 3)
    >>> # Load custom attribute
    >>> attributes = load_nx('graph.pkl', key='custom_attribute')

    Notes
    -----
    The NetworkX graph must have nodes with the specified attribute.
    """
    import pickle
    with open(filename, mode='rb') as f:
        G = pickle.load(f)

    X = []
    for node in G.nodes:
        X.append(G.nodes[node][key])

    return np.array(X)


def save_json(filename, G, **kwargs):
    """Save a tree graph to a JSON file.

    This function exports a NetworkX graph to a JSON file format,
    storing both node positions and edge connections.

    Parameters
    ----------
    filename : str or pathlib.Path
        Path to the JSON file to write.
    G : networkx.Graph
        Graph to write. Nodes must have a 'position' attribute.
    **kwargs
        Additional keyword arguments to pass to ``json.dumps()``.
        If 'indent' is not provided, it defaults to ``2``.

    Examples
    --------
    >>> import networkx as nx
    >>> from skeleton_refinement.io import save_json
    >>> G = nx.Graph()
    >>> G.add_node(0, position=[0,0,0])
    >>> G.add_node(1, position=[1,1,1])
    >>> G.add_edge(0, 1)
    >>> save_json('graph.json', G)
    >>> # With custom JSON formatting
    >>> save_json('pretty_graph.json', G, indent=4, sort_keys=True)

    Notes
    -----
    The output JSON structure will contain:
    - 'points': list of node positions
    - 'lines': list of edges
    """
    import json
    data = {
        "points": [G.nodes[node]['position'] for node in G.nodes],
        "lines": list(G.edges),
    }
    if 'indent' not in kwargs:
        kwargs.update({'indent': 2})
    with open(filename, 'w') as f:
        f.writelines(json.dumps(data, **kwargs))
    return


def save_nx(filename, G, **kwargs):
    """Save a tree graph to a pickle file.

    This function saves a NetworkX graph to a pickle file for later retrieval.

    Parameters
    ----------
    filename : str or pathlib.Path
        Path to the pickle file to write.
    G : networkx.Graph
        Graph to write.
    **kwargs
        Additional keyword arguments to pass to pickle.dump().
        If 'protocol' is not provided, it defaults to pickle.HIGHEST_PROTOCOL.

    Examples
    --------
    >>> import networkx as nx
    >>> from skeleton_refinement.io import save_nx
    >>> G = nx.Graph()
    >>> G.add_node(0, position=[0,0,0])
    >>> G.add_node(1, position=[1,1,1])
    >>> G.add_edge(0, 1)
    >>> save_nx('graph.pkl', G)
    >>> # With specific protocol
    >>> save_nx('graph_v2.pkl', G, protocol=2)

    Notes
    -----
    The pickle format is not secure against erroneous or maliciously constructed data.
    Never unpickle data received from untrusted or unauthenticated sources.
    """
    import pickle
    if 'protocol' not in kwargs:
        kwargs['protocol'] = pickle.HIGHEST_PROTOCOL
    if isinstance(filename, str):
        filename = Path(filename)
    with filename.open(mode='wb') as f:
        pickle.dump(G, f, **kwargs)
    return
