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


def estimate_sigma2_memory(X, Y):
    """Calculate the estimated memory usage for sigma squared computation.

    This function estimates the required memory for the computation of sigma squared
    based on given input arrays. The memory estimation accounts for the creation of
    intermediate arrays (`XX`, `YY`, `diff`, and `err`), which are simultaneously
    needed during computations.

    Parameters
    ----------
    X : numpy.ndarray
        The first input array of shape (N, D), where N is the number of samples and
        D is the number of dimensions.
    Y : numpy.ndarray
        The second input array of shape (M, _), where M is the number of samples.

    Returns
    -------
    int
        The estimated number of bytes required for the computation of sigma squared.

    Examples
    --------
    >>> import numpy as np
    >>> from skeleton_refinement.utilities import estimate_sigma2_memory
    >>> from skeleton_refinement.utilities import human_readable_size
    >>> # Create large point sets
    >>> X = np.random.rand(100000, 3)  # a point-cloud of 100,000 points in 3D
    >>> Y = np.random.rand(1000, 3)  # a skeleton of 1,000 points in 3D
    >>> req_mem = estimate_sigma2_memory(X, Y)
    >>> print(req_mem)
    9600000000
    >>> print(human_readable_size(req_mem))
    8.94 GB
    """
    N, D = X.shape
    M, _ = Y.shape
    dtype_byte = X.itemsize
    # Multiply by 4 since we need to create 4 arrays (`XX`, `YY`, `diff` & `err`)
    n_bytes = dtype_byte * M * N * D * 4
    return n_bytes


def get_available_memory():
    """
    Gets the amount of available physical memory in bytes using the `psutil` library.

    This function utilizes the `psutil` library to retrieve the current available physical
    memory on the system. It returns the number of bytes of memory that can be allocated
    for new or existing processes without causing pagefile swapping.

    Returns
    -------
    int
        The amount of available memory in bytes.

    Examples
    --------
    >>> from skeleton_refinement.utilities import get_available_memory
    >>> from skeleton_refinement.utilities import human_readable_size
    >>> free_mem = get_available_memory()
    >>> print(free_mem)
    7247376384
    >>> print(human_readable_size(free_mem))
    6.75 GB
    """
    import psutil
    return psutil.virtual_memory().available


def human_readable_size(bytes_value):
    """Convert a size in bytes to a human-readable string with appropriate unit.

    Parameters
    ----------
    bytes_value : int
        Size in bytes

    Returns
    -------
    str
        Human-readable string with size and unit (B, KB, MB, GB, TB)
    """
    # Define size units
    units = ['B', 'KB', 'MB', 'GB', 'TB', 'PB']

    # Use logarithmic approach to determine the appropriate unit
    if bytes_value == 0:
        return "0 B"

    # Find the appropriate unit
    unit_index = 0
    while bytes_value >= 1024 and unit_index < len(units) - 1:
        bytes_value /= 1024
        unit_index += 1

    # Format with 2 decimal places if not in bytes
    if unit_index == 0:
        return f"{int(bytes_value)} {units[unit_index]}"
    else:
        return f"{bytes_value:.2f} {units[unit_index]}"


def initialize_sigma2_fast(X, Y):
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
    >>> from skeleton_refinement.utilities import initialize_sigma2_fast
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


def initialize_sigma2_mem_efficient(X, Y, batch_size=1000):
    """Memory-efficient version of initialize_sigma2 that processes the point cloud in batches.

    Parameters
    ----------
    X : numpy.ndarray
        First point set of shape ``(N, D)`` where ``N`` is the number of point-cloud points and
        ``D`` is the dimensionality of each point.
    Y : numpy.ndarray
        Second point set of shape ``(M, D)`` where ``M`` is the number of skeleton points and
        ``D`` is the dimensionality of each point.
    batch_size : int, optional
        The size of the batch to process at a time, that is, the number of point-cloud points.
        Defaults to ``1000``.

    Returns
    -------
    float
        The initial variance value, calculated as the mean squared distance
        between all point pairs from `X` and `Y`.

    Examples
    --------
    >>> import numpy as np
    >>> from skeleton_refinement.utilities import initialize_sigma2_mem_efficient
    >>> from skeleton_refinement.utilities import estimate_sigma2_memory
    >>> # Create sample point sets
    >>> X = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])  # 3 points in 3D
    >>> Y = np.array([[0.1, 0.1, 0.1], [1.1, 0.1, 0.1]])  # 2 points in 3D
    >>> sigma2 = initialize_sigma2_mem_efficient(X, Y)
    >>> print(f"{sigma2:.6f}")
    0.276667
    >>> # Create large point sets
    >>> X = np.random.rand(100000, 3)  # a point-cloud of 100,000 points in 3D
    >>> Y = np.random.rand(1000, 3)  # a skeleton of 1,000 points in 3D
    >>> sigma2 = initialize_sigma2_mem_efficient(X, Y)
    >>> print(f"{sigma2:.6f}")
    0.165825
    """
    N, D = X.shape  # N = number of points in X, D = dimensionality of points
    M, _ = Y.shape  # M = number of points in Y

    # Initialize sum of squared differences
    sum_sq_diff = 0.0

    # Process X points in batches to reduce memory usage
    batch_size = min(batch_size, N)  # Adjust batch size based on available memory

    for i in range(0, N, batch_size):
        batch_end = min(i + batch_size, N)
        batch_X = X[i:batch_end]

        # For each point in Y, compute distance to all points in the current X batch
        for j in range(M):
            # Get current Y point and reshape to (1, D) for broadcasting
            y_point = Y[j].reshape(1, D)

            # Calculate squared distances between all batch_X points and this Y point
            diff = batch_X - y_point
            sum_sq_diff += np.sum(diff * diff)

    # Return the mean squared distance
    return sum_sq_diff / (D * M * N)


def initialize_sigma2(X, Y, batch_size=1000):
    """Initialize the variance parameter for point set registration algorithms.

    Selects and initializes the appropriate method for computing `sigma2` based on
    the available system memory and the estimated memory requirement for processing
    input data. Switches to a memory-efficient method if the required memory
    exceeds the available system memory.

    Parameters
    ----------
    X : array-like
        Input data, typically representing the feature matrix.
    Y : array-like
        Target data, typically representing an outcome or dependent variable.
    batch_size : int, optional
        Batch size for the memory-efficient implementation.
        Defaults to ``1000``.

    Returns
    -------
    float
        The initial variance value, calculated as the mean squared distance
        between all point pairs from `X` and `Y`.
    """
    required_memory = estimate_sigma2_memory(X, Y)
    available_memory = get_available_memory()
    if required_memory < available_memory:
        return initialize_sigma2_fast(X, Y)
    else:
        return initialize_sigma2_mem_efficient(X, Y, batch_size=batch_size)


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
