#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
## Point Set Registration Utilities

This module provides efficient mathematical functions to support point set registration algorithms, particularly for the Coherent Point Drift (CPD) method used in 3D point cloud alignment and deformable registration tasks.

### Key Features

- Variance initialization for point sets using efficient vectorized operations
- Gaussian kernel matrix computation for smooth spatial transformations
- Optimization for large point clouds through NumPy vectorization
- Support for N-dimensional point data (typically 2D or 3D spatial coordinates)
"""

import numpy as np


def initialize_sigma2(X, Y):
    """Initialize the variance parameter for point set registration algorithms.

    This function calculates the initial variance (sigma squared) between two point sets,
    which is used as a normalization factor in point set registration algorithms.
    It computes the mean squared distance between all pairs of points from sets X and Y.

    Parameters
    ----------
    X : numpy.ndarray
        First point set of shape ``(N, D)`` where ``N`` is the number of points and
        ``D`` is the dimensionality of each point (typically 2 or 3 for spatial coordinates).
    Y : numpy.ndarray
        Second point set of shape ``(M, D)`` where ``M`` is the number of points and
        ``D`` is the dimensionality of each point (should match the dimensionality of X).

    Returns
    -------
    float
        The initial variance value, calculated as the mean squared distance
        between all point pairs from `X` and `Y`.

    Notes
    -----
    This function is typically used in point set registration algorithms like
    Coherent Point Drift (CPD) to initialize the variance parameter.

    The computation avoids explicit loops by using broadcasting and tiling operations
    to efficiently calculate distances between all point pairs.

    Examples
    --------
    >>> import numpy as np
    >>> from skeleton_refinement.utilities import initialize_sigma2
    >>> X = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])  # 3 points in 3D
    >>> Y = np.array([[0.1, 0.1, 0.1], [1.1, 0.1, 0.1]])  # 2 points in 3D
    >>> sigma2 = initialize_sigma2(X, Y)
    >>> print(f"{sigma2:.6f}")
    0.276667
    """
    # Get dimensions of the point sets
    (N, D) = X.shape  # N = number of points in X, D = dimensionality of points
    (M, _) = Y.shape  # M = number of points in Y

    # Reshape X to add a dimension for broadcasting (1×N×D)
    XX = np.reshape(X, (1, N, D))
    # Reshape Y to add a dimension for broadcasting (M×1×D)
    YY = np.reshape(Y, (M, 1, D))
    # Replicate XX along M dimension to create matrix of size (M×N×D)
    XX = np.tile(XX, (M, 1, 1))
    # Replicate YY along N dimension to create matrix of size (M×N×D)
    YY = np.tile(YY, (1, N, 1))

    # Calculate the difference between every pair of points
    diff = XX - YY
    # Square the differences
    err = np.multiply(diff, diff)

    # Return the mean squared distance between all point pairs
    # Normalized by dimension D and total number of point pairs (M*N)
    return np.sum(err) / (D * M * N)


def gaussian_kernel(Y, beta):
    """Compute the Gaussian kernel matrix for a point set.

    This function calculates a Gaussian kernel matrix (Gram matrix) where each element
    ``(i,j)`` represents the Gaussian radial basis function between points ``Y[i]`` and ``Y[j]``.
    The kernel is used in deformable point set registration algorithms to define
    smooth spatial transformations.

    Parameters
    ----------
    Y : numpy.ndarray
        Point set of shape ``(M, D)`` where ``M`` is the number of points and
        ``D`` is the dimensionality of each point (typically 2 or 3 for spatial coordinates).
    beta : float
        The width parameter controlling the spatial scale of the Gaussian kernel.
        Larger values result in smoother transformations but less accurate registrations.

    Returns
    -------
    numpy.ndarray
        Gaussian kernel matrix of shape ``(M, M)``, where each element ``(i,j)`` is:
        ``exp(-||Y[i] - Y[j]||^2 / (2 * beta))``

    Notes
    -----
    This function is typically used in deformable registration algorithms like
    Coherent Point Drift (CPD) to define smooth transformations between point sets.

    The implementation uses broadcasting and tiling operations to efficiently
    compute all pairwise distances without explicit loops.

    Examples
    --------
    >>> import numpy as np
    >>> from skeleton_refinement.utilities import gaussian_kernel
    >>> Y = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])  # 3 points in 3D
    >>> beta = 1.0
    >>> G = gaussian_kernel(Y, beta)
    >>> print(G.shape)
    (3, 3)
    >>> print(np.round(G, 3))
    [[1.    0.607 0.607]
     [0.607 1.    0.368]
     [0.607 0.368 1.   ]]
    """
    # Extract dimensions from input point set
    (M, D) = Y.shape  # M = number of points, D = dimensions per point

    # Reshape Y into a 3D tensor with shape (1, M, D) for broadcasting
    XX = np.reshape(Y, (1, M, D))
    # Reshape Y into a 3D tensor with shape (M, 1, D) for broadcasting
    YY = np.reshape(Y, (M, 1, D))
    # Tile XX to create a tensor of shape (M, M, D) where each "row" is repeated M times
    XX = np.tile(XX, (M, 1, 1))
    # Tile YY to create a tensor of shape (M, M, D) where each "column" is repeated M times
    YY = np.tile(YY, (1, M, 1))

    # Calculate the element-wise difference between each pair of points
    diff = XX - YY
    # Square the differences
    diff = np.multiply(diff, diff)
    # Sum along the dimension axis to get squared Euclidean distances between each pair of points
    diff = np.sum(diff, 2)

    # Apply Gaussian RBF to get the kernel matrix: exp(-||Y[i] - Y[j]||^2 / (2 * beta))
    return np.exp(-diff / (2 * beta))
