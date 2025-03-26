#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
## Stochastic Point Cloud Registration

A module for aligning and registering 3D skeleton structures to point clouds using stochastic optimization techniques.
It enables precise fitting of skeletal models to noisy or dense point cloud data, commonly used in 3D plant structure analysis.

### Key Features

- Non-rigid point set registration using Coherent Point Drift (CPD) algorithm
- Optimized skeleton alignment to underlying point cloud structures
- Minimum spanning tree construction from point sets using k-nearest neighbors
- Configurable regularization parameters for controlling deformation smoothness
- Support for handling outliers and varying point densities

### Usage Examples

```python
>>> from skeleton_refinement.stochastic_registration import perform_registration, knn_mst
>>> from skeleton_refinement.io import load_ply, load_json
>>> # Load point cloud and skeleton data
>>> pcd = load_ply("path/to/pointcloud.ply")
>>> skel = load_json("path/to/skeleton.json", "points")
>>> # Perform registration to align skeleton with point cloud
>>> refined_skel = perform_registration(pcd, skel, alpha=5, beta=5)
>>> # Generate skeleton graph structure
>>> skel_tree = knn_mst(refined_skel)
```
"""

import networkx as nx
import numpy as np
from sklearn.neighbors import NearestNeighbors

from skeleton_refinement.deformable_registration import DeformableRegistration
from skeleton_refinement.utilities import initialize_sigma2


def perform_registration(X, Y, **kwargs):
    """Performs stochastic deformation registration to align a skeleton to a point cloud.

    This function uses the Coherent Point Drift (CPD) algorithm to perform non-rigid
    registration between a reference point cloud and a skeleton. The algorithm optimizes
    the skeleton's position to better align with the underlying point cloud structure.

    Parameters
    ----------
    X : numpy.ndarray
        The input reference point cloud coordinates of shape `(n_points, dim)`, XYZ sorted.
        This represents the fixed point set that the skeleton will be aligned to.
    Y : numpy.ndarray
        The input reference skeleton coordinates of shape `(n_points, dim)`, XYZ sorted.
        This represents the moving point set that will be transformed.

    Other Parameters
    ----------------
    alpha : float
        Regularization weight that controls the smoothness of deformation.
        Higher values result in smoother, more rigid transformations.
    beta : float
        Width of the Gaussian kernel used in the non-rigid transformation.
        Controls the influence range of each point in the deformation.
    sigma2 : float or numpy.ndarray, optional
        Initial variance of the Gaussian Mixture Model.
        If None, it will be estimated automatically from the point sets.
        Defaults to ``None``.
    max_iterations : int, optional
        The maximum number of iterations before stopping the iterative registration.
        Defaults to ``100``.
    tolerance : float, optional
        Convergence criterion. The algorithm stops when the error falls below this value.
        Defaults to ``0.001``.
    w : float, optional
        Weight of the uniform distribution in the mixture model.
        Used to account for outliers. Value between 0 and 1.
        Defaults to ``0``.

    Returns
    -------
    numpy.ndarray
        The transformed skeleton coordinates of shape `(n_points, dim)`, XYZ sorted.
        This is the optimized skeleton aligned to the input point cloud.

    Notes
    -----
    The function internally uses the DeformableRegistration class which implements
    the Coherent Point Drift algorithm. The registration process involves an
    Expectation-Maximization approach to optimize the transformation parameters.

    The parameters alpha and beta are crucial for obtaining good results:
    - Higher alpha values enforce more rigidity in the transformation
    - Beta determines the spatial extent of interaction between points

    See Also
    --------
    skeleton_refinement.deformable_registration.DeformableRegistration : Class that implements the CPD algorithm.
    skeleton_refinement.expectation_maximization_registration.ExpectationMaximizationRegistration : Base class for EM-based registration.

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
    # Add input point sets to kwargs to pass to DeformableRegistration
    kwargs.update({'X': X, 'Y': Y})

    # Initialize the Coherent Point Drift registration object
    reg = DeformableRegistration(**kwargs)
    # Perform initial transformation of the point cloud
    reg.transform_point_cloud()

    # If sigma2 (variance of GMM) is not provided, estimate it and run EM iterations
    if reg.sigma2 is None:
        # Initialize sigma2 based on current point sets
        reg.sigma2 = initialize_sigma2(reg.X, reg.TY)
        # Update objective function with new sigma2 value
        # (q is the log-likelihood with regularization terms)
        reg.q = -reg.err - reg.N * reg.D / 2 * np.log(reg.sigma2)
        # Iteratively refine transformation until convergence or max iterations reached
        while reg.iteration < reg.max_iterations and reg.err > reg.tolerance:
            reg.iterate()  # Run one EM iteration

    return reg.TY


def knn_mst(skeleton_points, n_neighbors=5, knn_algorithm='kd_tree', mst_algorithm='kruskal'):
    """Create a minimum spanning tree from skeleton points using k-nearest neighbors graph.

    This function constructs a k-nearest neighbors graph from the input skeleton points
    using Euclidean distances, then computes the minimum spanning tree of this graph.
    The resulting tree represents the skeleton structure as a connected graph with
    minimal total edge weight.

    Parameters
    ----------
    skeleton_points : numpy.ndarray
        The skeleton coordinates of shape ``(n_points, 3)``, with XYZ coordinates.
        Each row represents the 3D position of a skeleton point.
    n_neighbors : int, optional
        The number of neighbors to consider for each point when building the
        k-nearest neighbors graph. Default is ``5``.
    knn_algorithm : {'auto', 'ball_tree', 'kd_tree', 'brute'}, optional
        The algorithm to use for computing the k-nearest neighbors.
        - 'kd_tree': KD Tree algorithm, works well for low dimensions.
        - 'ball_tree': Ball Tree algorithm, works well for high dimensions.
        - 'brute': Brute force algorithm, always uses all data points.
        - 'auto': Automatically chooses the most appropriate algorithm.
        Default is `'kd_tree'`.
    mst_algorithm : {'kruskal', 'prim', 'boruvka'}, optional
        The algorithm to use for computing the minimum spanning tree.
        - 'kruskal': Kruskal's algorithm, efficient for sparse graphs.
        - 'prim': Prim's algorithm, efficient for dense graphs.
        - 'boruvka': Borůvka's algorithm, another MST algorithm.
        Default is `'kruskal'`.

    Returns
    -------
    networkx.Graph
        A NetworkX Graph representing the skeleton structure as a minimum spanning tree.
        - Nodes correspond to skeleton points with their 3D coordinates stored as a 'position' attribute
        - Edges connect points that are part of the minimum spanning tree
        - Edge weights represent Euclidean distances between connected points

    Notes
    -----
    The function first builds a k-nearest neighbors graph where each point is connected
    to its k nearest neighbors. Then it computes the minimum spanning tree of this graph
    to create a connected structure with minimal total edge length.

    The Minkowski distance with ``p=2`` is used, which corresponds to the Euclidean distance.

    See Also
    --------
    sklearn.neighbors.NearestNeighbors : For finding nearest neighbors
    networkx.minimum_spanning_tree : For computing minimum spanning trees

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
    >>> # The output is a NetworkX Graph with nodes representing skeleton points
    >>> print(f"Number of nodes: {skel_tree.number_of_nodes()}")
    >>> print(f"Number of edges: {skel_tree.number_of_edges()}")
    >>> # Access node coordinates
    >>> sample_node = list(skel_tree.nodes())[0]
    >>> print(f"Position of node {sample_node}: {skel_tree.nodes[sample_node]['position']}")
    """
    # Initialize nearest neighbors model with Euclidean distance (Minkowski p=2)
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm=knn_algorithm, metric="minkowski", p=2).fit(
        skeleton_points)
    # Find k-nearest neighbors for each point - returns distances and indices matrices
    distances, indices = nbrs.kneighbors(skeleton_points)

    # Initialize empty undirected graph to build KNN representation
    G = nx.Graph()

    # Add edges between each point and its k-nearest neighbors
    for row, nodes_idx in enumerate(indices):
        nodes_idx = list(map(int, nodes_idx))
        node_idx, nei_idx = nodes_idx[0], nodes_idx[1:]  # First index is the point itself
        # Add edges with Euclidean distances as weights
        # Skip first neighbor (index 0) as it's the point itself
        [G.add_edges_from([(node_idx, n_idx, {"weight": distances[row, col + 1]})]) for col, n_idx in
         enumerate(nei_idx)]

    # Store original 3D coordinates as node attributes
    for node_id in G.nodes:
        G.nodes[node_id]['position'] = skeleton_points[node_id]

    # Compute minimum spanning tree from the KNN graph
    # This creates optimal connected structure with minimal total edge length
    T = nx.minimum_spanning_tree(G, algorithm=mst_algorithm)

    return T
