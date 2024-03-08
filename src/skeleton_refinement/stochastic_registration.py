#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This code implements the basic structure of for performing the stochastic
optimization algorithm. Given two set of discrete points, this code returns
the transformed point set.
"""
import networkx as nx
import numpy as np
from sklearn.neighbors import NearestNeighbors

from skeleton_refinement.deformable_registration import DeformableRegistration
from skeleton_refinement.utilities import initialize_sigma2


def perform_registration(X, Y, **kwargs):
    """Performs the skeleton optimization using stochastic deformation registration.

    Parameters
    ----------
    X : numpy.ndarray
        The input reference point cloud coordinates of shape `(n_points, dim)`, XYZ sorted.
    Y : numpy.ndarray
        The input reference skeleton coordinates of shape `(n_points, dim)`, XYZ sorted.

    Other Parameters
    ----------------
    alpha : float
        ???.
    beta : float
        ???.
    sigma2 : numpy.ndarray, optional
        ???
        Defaults to `None`.
    max_iterations : int, optional
        The maximum number of iterations before stopping the iterative registration.
        Defaults to `100`.
    tolerance : float, optional
        ??? Tolerance for registration.
        Defaults to `0.001`.
    w : int, optional
        ???
        Defaults to `0`.

    Returns
    -------
    numpy.ndarray
        The transformed skeleton coordinates of shape `(n_points, 3)`, XYZ sorted.

    Examples
    --------
    >>> from skeleton_refinement.stochastic_registration import perform_registration
    >>> from skeleton_refinement.io import load_ply, load_json
    >>> pcd = load_ply("real_plant_analyzed/PointCloud_1_0_1_0_10_0_7ee836e5a9/PointCloud.ply")
    >>> skel = load_json("real_plant_analyzed/CurveSkeleton__TriangleMesh_0393cb5708/CurveSkeleton.json", "points")
    >>> # Perform stochastic optimization
    >>> refined_skel = perform_registration(pcd, skel, alpha=5, beta=5)
    >>> print(refined_skel.shape)
    """
    kwargs.update({'X': X, 'Y': Y})
    reg = DeformableRegistration(**kwargs)
    reg.transform_point_cloud()
    if reg.sigma2 is None:
        reg.sigma2 = initialize_sigma2(reg.X, reg.TY)
        reg.q = -reg.err - reg.N * reg.D / 2 * np.log(reg.sigma2)
        while reg.iteration < reg.max_iterations and reg.err > reg.tolerance:
            reg.iterate()
    return reg.TY


def knn_mst(skeleton_points, n_neighbors=5, knn_algorithm='kd_tree', mst_algorithm='kruskal'):
    """Update the skeleton structure with minimum spanning tree on knn-graph with Euclidean distances.

    Parameters
    ----------
    skeleton_points : numpy.ndarray
        The skeleton coordinates of shape `(n_points, 3)`, XYZ sorted.
    n_neighbors : int, optional
        The number of neighbors to search for in `skeleton_points`.
        Default is `5`.
    knn_algorithm : str, optional
        The algorithm to use for computing the kNN distance.
        Must be one of 'auto', 'ball_tree', 'kd_tree' or 'brute'.
        Defaults to `kd_tree`.
    mst_algorithm : str, optional
        The algorithm to use for computing the minimum spanning tree.
        Must be one of 'kruskal', 'prim' or 'boruvka'.
        Defaults to `kruskal`.

    See Also
    --------
    sklearn.neighbors.NearestNeighbors
    networkx.minimum_spanning_tree

    Returns
    -------
    networkx.Graph
        The skeleton structure with minimum spanning tree from knn-graph.

    Examples
    --------
    >>> from skeleton_refinement.stochastic_registration import perform_registration
    >>> from skeleton_refinement.stochastic_registration import knn_mst
    >>> from skeleton_refinement.io import load_ply, load_json
    >>> pcd = load_ply("real_plant_analyzed/PointCloud_1_0_1_0_10_0_7ee836e5a9/PointCloud.ply")
    >>> skel = load_json("real_plant_analyzed/CurveSkeleton__TriangleMesh_0393cb5708/CurveSkeleton.json", "points")
    >>> # Perform stochastic optimization
    >>> refined_skel = perform_registration(pcd, skel, alpha=5, beta=5)
    >>> # Compute skeleton tree structure using mst on knn-graph:
    >>> skel_tree = knn_mst(refined_skel)
    """
    # Find the k-nearest neighbors:
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm=knn_algorithm, metric="minkowski", p=2).fit(skeleton_points)
    distances, indices = nbrs.kneighbors(skeleton_points)

    # - Create a k-neighbors graph with paired nodes and Euclidean distances as edge weights:
    G = nx.Graph()
    # -- Add the edges and the weight:
    for row, nodes_idx in enumerate(indices):
        node_idx, nei_idx = nodes_idx[0], nodes_idx[1:]
        [G.add_edges_from([(node_idx, n_idx, {"weight": distances[row, col + 1]})]) for col, n_idx in
         enumerate(nei_idx)]
    # -- Add the node coordinates:
    for node_id in G.nodes:
        G.nodes[node_id]['position'] = skeleton_points[node_id]
    # -- Find the minimum spanning tree:
    T = nx.minimum_spanning_tree(G, algorithm=mst_algorithm)

    return T