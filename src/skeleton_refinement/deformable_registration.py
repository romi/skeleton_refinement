#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
## Deformable Registration

This module provides a non-rigid point set registration implementation using the Coherent Point Drift (CPD) algorithm, enabling accurate alignment of point clouds with non-linear deformations.

### Key Features

- Non-rigid point cloud alignment using Gaussian Mixture Models (GMM)
- Coherent Point Drift algorithm implementation with customizable parameters
- Efficient transformation calculation with regularization for smooth deformations
- Support for arbitrary dimensional point clouds
- Built on a generic Expectation-Maximization registration framework

### Notes

This is a part of the implementation of the stochastic registration algorithm based on the following paper:
Myronenko A. and Song X. (2010) **Point set registration: Coherent Point drift**.
_IEEE Transactions on Pattern Analysis and Machine Intelligence_. 32 (2): 2262-2275.
DOI: [10.1109/TPAMI.2010.46](https://doi.org/10.1109/TPAMI.2010.46)

The library is based on the python implementation of the paper in ``pycpd`` package.

### Usage Examples

```python
>>> import numpy as np
>>> from skeleton_refinement.deformable_registration import DeformableRegistration
>>> # Create sample point sets
>>> X = np.random.rand(10, 3)  # Reference point set
>>> Y = np.random.rand(10, 3)  # Point set to be aligned
>>> # Initialize and run registration
>>> reg = DeformableRegistration(X=X, Y=Y, alpha=2, beta=2)
>>> TY = reg.register()
>>> # Get registration parameters
>>> G, W = reg.get_registration_parameters()
```
"""

import numpy as np

from skeleton_refinement.expectation_maximization_registration import ExpectationMaximizationRegistration
from skeleton_refinement.utilities import gaussian_kernel

ALPHA = 2  # default value of alpha
BETA = 2  # default value of beta


class DeformableRegistration(ExpectationMaximizationRegistration):
    """Deformable point set registration using Coherent Point Drift algorithm.

    This class implements the non-rigid point set registration algorithm from the paper:
    "Point Set Registration: Coherent Point Drift" by Myronenko and Song (2010).
    It optimizes a Gaussian Mixture Model (GMM) to find correspondences between
    two point sets and computes a non-rigid transformation.

    Attributes
    ----------
    alpha : float
        Regularization weight controlling the smoothness of deformation.
    beta : float
        Width of Gaussian kernel used in the non-rigid transformation.
    W : numpy.ndarray
        Deformation matrix of shape ``(M, D)`` where ``M`` is the number of points in ``Y``
        and ``D`` is the dimension.
    G : numpy.ndarray
        Gaussian kernel matrix of shape ``(M, M)``, computed from ``Y`` using `beta` as width.
    TY : numpy.ndarray
        The transformed point set ``Y`` after registration.
    sigma2 : float
        Final variance of GMM.

    Notes
    -----
    The implementation uses Expectation-Maximization algorithm to optimize the transformation.
    The non-rigid transformation is represented as ``T(Y) = Y + G*W`` where ``G`` is the Gaussian kernel and ``W`` is optimized.

    See Also
    --------
    skeleton_refinement.expectation_maximization_registration.ExpectationMaximizationRegistration : Base class for EM-based registration algorithms.

    Examples
    --------
    >>> import numpy as np
    >>> from skeleton_refinement.deformable_registration import DeformableRegistration
    >>> # Create sample point sets
    >>> X = np.random.rand(10, 3)  # Reference point set
    >>> Y = np.random.rand(10, 3)  # Point set to be aligned
    >>> # Initialize and run registration
    >>> reg = DeformableRegistration(X=X, Y=Y, alpha=2, beta=2)
    >>> TY = reg.register()
    >>> # Get registration parameters
    >>> G, W = reg.get_registration_parameters()
    """

    def __init__(self, alpha=ALPHA, beta=BETA, *args, **kwargs):
        """Initialize the deformable registration algorithm.

        Parameters
        ----------
        alpha : float, optional
            Regularization weight controlling the smoothness of deformation.
            Higher values result in smoother deformation. Default is ``2``.
        beta : float, optional
            Width of Gaussian kernel used in the non-rigid transformation.
            Controls the interaction between points. Default is ``2``.
        X : numpy.ndarray
            Reference point set of shape ``(N, D)`` where ``N`` is number of points and ``D`` is dimension.
        Y : numpy.ndarray
            Point set to be aligned to ``X``, of shape ``(M, D)`` where ``M`` is number of points.
        sigma2 : float, optional
            Initial variance of GMM. If ``None``, it's computed from data.
        max_iterations : int, optional
            Maximum number of iterations for the optimization algorithm.
        tolerance : float, optional
            Convergence threshold based on change in `sigma2`.
        w : float, optional
            Weight of the uniform distribution component, range ``[0,1]``.
            Used to account for outliers. Default is ``0``.
        """
        super().__init__(*args, **kwargs)
        self.alpha = ALPHA if alpha is None else alpha
        self.beta = BETA if alpha is None else beta
        self.W = np.zeros((self.M, self.D))
        self.G = gaussian_kernel(self.Y, self.beta)

    def update_transform(self):
        """Update the transformation parameters.

        Solves for the deformation matrix W that minimizes the energy function.
        This is computed by solving the linear system: ``(DP1*G + alpha*sigma2*I)*W = P*X - DP1*Y``, where:

          - ``DP1`` is a diagonal matrix with elements of ``P1``,
          - ``G`` is the Gaussian kernel,
          - ``I`` is the identity matrix,
          - ``P`` is the posterior probability matrix.
        """
        # Solve for optimal deformation matrix W in CPD algorithm
        # A: Left side of linear equation system combining point correspondences and regularization
        A = np.dot(np.diag(self.P1), self.G) + self.alpha * self.sigma2 * np.eye(self.M)  # P1-weighted kernel matrix + regularization term

        # B: Right side of equation system representing the difference between points
        B = np.dot(self.P, self.X) - np.dot(np.diag(self.P1), self.Y)  # P-weighted X points minus P1-weighted Y points

        # Compute deformation matrix W by solving linear system AW = B
        self.W = np.linalg.solve(A, B)  # W determines how points in Y are transformed

    def transform_point_cloud(self, Y=None):
        """Apply the non-rigid transformation to a point cloud.

        The transformation is defined as: ``T(Y) = Y + G*W``,
        where ``G`` is the Gaussian kernel and ``W`` is the deformation matrix.

        Parameters
        ----------
        Y : numpy.ndarray, optional
            Point cloud to transform of shape ``(M, D)``.
            If `None`, transforms the stored point cloud ``self.Y``.

        Returns
        -------
        numpy.ndarray or None
            Transformed point cloud of the same shape as input ``Y``.
            If ``Y`` is ``None``, updates ``self.TY`` and returns `None`.
        """
        if Y is None:
            # Apply non-rigid transformation to the class's own point cloud
            # TY = Y + G*W where G is the Gaussian kernel matrix and W is the deformation matrix
            self.TY = self.Y + np.dot(self.G, self.W)
            return
        else:
            # Apply transformation to the input point cloud and return the result
            # Returns the transformed points without modifying internal state
            return Y + np.dot(self.G, self.W)

    def update_variance(self):
        """Update the variance (``sigma2``) of the Gaussian Mixture Model.

        Computes the weighted distance between the transformed ``Y`` (``TY``) and the
        reference point cloud ``X``, normalized by the number of points and dimensions.
        The updated variance is used to evaluate convergence in the EM algorithm.
        """
        # Store previous sigma2 value to calculate change later
        qprev = self.sigma2

        # Calculate weighted sum of squared norms of X points: P^T * (X^2)
        xPx = np.dot(np.transpose(self.Pt1), np.sum(np.multiply(self.X, self.X), axis=1))
        # Calculate weighted sum of squared norms of transformed Y points: P1^T * (TY^2)
        yPy = np.dot(np.transpose(self.P1), np.sum(np.multiply(self.TY, self.TY), axis=1))
        # Calculate trace of P * X * Y^T (cross-correlation term)
        trPXY = np.sum(np.multiply(self.TY, np.dot(self.P, self.X)))

        # Update sigma2 using the formula from CPD algorithm:
        # σ² = (xPx - 2*trPXY + yPy) / (Np * D)
        # where Np is number of points and D is dimensionality
        self.sigma2 = (xPx - 2 * trPXY + yPy) / (self.Np * self.D)

        # Prevent numerical issues by setting a minimum threshold for sigma2
        if self.sigma2 <= 0:
            self.sigma2 = self.tolerance / 10

        # Calculate absolute change in sigma2 for convergence check
        self.err = np.abs(self.sigma2 - qprev)

    def get_registration_parameters(self):
        """Retrieve the registration parameters `G` & `W`.

        Returns
        -------
        numpy.ndarray
            Gaussian kernel matrix of shape ``(M, M)``.
        numpy.ndarray
            Deformation matrix of shape ``(M, D)``.

        Notes
        -----
        These parameters can be used to apply the learned transformation
        to other point sets using the formula: ``Y_transformed = Y + G*W``
        """
        return self.G, self.W
