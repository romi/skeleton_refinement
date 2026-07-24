#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
# Deformable Registration Module

A Python implementation of the Coherent Point Drift (CPD) deformable registration algorithm, which aligns a moving point set to a reference set using an Expectation‑Maximization framework.
It provides a smooth, non‑rigid transformation that is useful for tasks such as 3D shape alignment, skeleton refinement, and point‑cloud registration.

## Key Features

- **EM‑based non‑rigid registration**: iteratively optimizes a Gaussian mixture model to fit the source to the target.
- **Configurable regularization**: `alpha` controls the trade‑off between data fidelity and smoothness, while `beta` sets the Gaussian kernel width.
- **Deformation matrix (`W`) and kernel matrix (`G`)**: accessible for inspection or custom post‑processing.
- **Utility methods**
  - `update_transform()`: solves for the deformation matrix.
  - `transform_point_cloud()`: applies the current deformation to any point cloud.
  - `update_variance()`: recomputes the mixture‑model variance to maintain numerical stability.
  - `get_registration_parameters()`: returns the internal `G` and `W` matrices.

## Usage Examples

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

## Reference

This is a part of the implementation of the stochastic registration algorithm based on the following paper:
Myronenko A. and Song X. (2010) **Point set registration: Coherent Point drift**.
_IEEE Transactions on Pattern Analysis and Machine Intelligence_. 32 (2): 2262-2275.
DOI: [10.1109/TPAMI.2010.46](https://doi.org/10.1109/TPAMI.2010.46)
arXiv [PDF](https://arxiv.org/pdf/0905.2635).

The library is based on the Python implementation of the paper in ``pycpd`` package.
[GitHub](https://github.com/siavashk/pycpd) sources.
[PyPi](https://pypi.org/project/pycpd/) package.
"""

import numpy as np

from skeleton_refinement.expectation_maximization_registration import ExpectationMaximizationRegistration
from skeleton_refinement.utilities import gaussian_kernel
from skeleton_refinement.utilities import initialize_sigma2

ALPHA = 2  # regularization weight controlling smoothness of deformation
BETA = 2  # Gaussian kernel width governing the kernel's influence radius


class DeformableRegistration(ExpectationMaximizationRegistration):
    """
    Implement a deformable point‑set registration using the Expectation‑Maximization
    framework described in Myronenko & Song (2010).

    The algorithm treats the moving point set as a Gaussian mixture model and iteratively
    updates a smooth, non‑rigid transformation that aligns it to the reference set.
    Regularization is controlled by the ``alpha`` parameter, while ``beta`` determines
    the width of the Gaussian kernel used to enforce smoothness.

    Attributes
    ----------
    alpha: float
        Trade‑off between the likelihood term and the regularization term.
        Must be a positive value; larger values enforce a smoother transformation.
    beta: float
        Width of the Gaussian kernel (variance of the smoothing kernel).
        Must be a positive value; larger values produce a broader influence region.
    W: numpy.ndarray, shape (M, D)
        Deformable transformation matrix that maps the reference points to the target space.
        Updated at each EM iteration.
    G: numpy.ndarray, shape (M, M)
        Pre‑computed Gaussian kernel matrix between all points in the reference set ``self.Y``.
        Used to express the smoothness constraint.
    """

    def __init__(self, alpha=ALPHA, beta=BETA, *args, **kwargs):
        """
        Initialize the deformable registration algorithm.

        Parameters
        ----------
        alpha: float, optional
            Trade‑off between the maximum‑likelihood fit and the regularization term.
            Must be positive. Defaults to ``2`` (the value of ``ALPHA``).
        beta: float, optional
            Width of the Gaussian kernel. Must be positive. Defaults to ``2`` (the value of ``BETA``).
        X: numpy.ndarray
            Reference point set of shape ``(N, D)`` where ``N`` is the number of points and ``D`` is the dimension.
        Y: numpy.ndarray
            Point set to be aligned to ``X``, of shape ``(M, D)`` where ``M`` is the number of points.
        sigma2: float, optional
            Initial variance of GMM. If ``None``, it's computed from data.
        max_iterations: int, optional
            Maximum number of iterations for the optimization algorithm.
        tolerance: float, optional
            Convergence threshold based on change in `sigma2`.
        w: float, optional
            Weight of the uniform distribution component, range ``[0,1]``.
            Used to account for outliers. Default is ``0``.

        Notes
        -----
        The Gaussian kernel matrix ``G`` is computed once at construction time from the source points ``self.Y``
        and the provided ``beta``.

        Examples
        --------
        >>> import numpy as np
        >>> from skeleton_refinement.deformable_registration import DeformableRegistration
        >>> X = np.random.rand(20, 3)   # target point set
        >>> Y = np.random.rand(20, 3)   # source point set
        >>> reg = DeformableRegistration(alpha=2.0, beta=2.0, X=X, Y=Y)
        >>> reg.W.shape
        (20, 3)
        """
        super().__init__(*args, **kwargs)
        self.alpha = alpha if alpha else ALPHA  # regularization weight controlling smoothness of deformation
        self.beta = beta if beta else BETA  # Gaussian kernel width governing the kernel's influence radius
        self.W = np.zeros((self.M, self.D))
        self.G = gaussian_kernel(self.Y, self.beta)

    def update_transform(self) -> None:
        """
        Compute a new estimate of the deformable transformation matrix ``W``.

        The update follows Equation 22 of the CPD paper (Myronenko & Song, 2010) and
        solves a linear system ``A·W = B`` where ``A`` incorporates the regularization
        term ``alpha`` and the current estimate of the variance ``sigma2``.

        Solves for the deformation matrix W that minimizes the energy function.
        This is computed by solving the linear system: ``(DP1*G + alpha*sigma2*I)*W = P*X - DP1*Y``, where:

          - ``DP1`` is a diagonal matrix with elements of ``P1``,
          - ``G`` is the Gaussian kernel,
          - ``I`` is the identity matrix,
          - ``P`` is the posterior probability matrix.

        Notes
        -----
        The matrix ``A`` is guaranteed to be positive‑definite when ``alpha`` and
        ``sigma2`` are positive, ensuring that ``numpy.linalg.solve`` succeeds.

        The method updates the internal attribute ``self.W`` in place.

        See Also
        --------
        transform_point_cloud : Apply the newly estimated transformation to a point cloud.

        References
        ----------
        https://arxiv.org/pdf/0905.2635.pdf.
        """
        # Initialise `sigma2` if ``None``
        if self.sigma2 is None:
            # Ensure we have a transformed source cloud (identity transform at start)
            if self.TY is None:
                # With zero deformation (W = 0) the transformed points equal the original source points
                self.TY = self.Y.copy()
            # Initialize variance from the current alignment
            self.sigma2 = initialize_sigma2(self.X, self.TY)

        # Solve for optimal deformation matrix W in CPD algorithm
        # A: Left side of linear equation system combining point correspondences and regularization
        # P1-weighted kernel matrix + regularization term
        A = np.dot(np.diag(self.P1), self.G) + self.alpha * self.sigma2 * np.eye(self.M)

        # B: Right side of equation system representing the difference between points
        # P-weighted X points minus P1-weighted Y points
        B = np.dot(self.P, self.X) - np.dot(np.diag(self.P1), self.Y)

        # Compute deformation matrix W by solving linear system AW = B
        self.W = np.linalg.solve(A, B)  # W determines how points in Y are transformed

    def transform_point_cloud(self, Y=None) -> None | np.ndarray:
        """
        Transform a point cloud using the current deformable transformation.

        The transformation is defined as: ``T(Y) = Y + G*W``,
        where ``G`` is the Gaussian kernel and ``W`` is the deformation matrix.

        Parameters
        ----------
        Y : numpy.ndarray, shape (N, D), optional
            Points to be transformed. If ``None`` (default), the method updates the
            internal transformed source cloud ``self.TY`` in place using ``self.Y``.

        Returns
        -------
        numpy.ndarray or None
            * If ``Y`` is ``None``, the method returns ``None`` and stores the transformed source points in ``self.TY``.
            * Otherwise, a new ``numpy.ndarray`` containing the transformed copy of ``Y`` is returned.

        Raises
        ------
        ValueError
            If the shape of ``Y`` is incompatible with the stored kernel matrix ``G``.

        Notes
        -----
        The transformation applied is ``Y + G·W`` where ``G`` is the Gaussian kernel
        matrix computed from the source points and ``W`` is the current deformation matrix.
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

    def update_variance(self) -> None:
        """
        Update the variance ``sigma2`` of the Gaussian mixture model using the current deformation estimate.

        Computes the weighted distance between the transformed ``Y`` (``TY``) and the
        reference point cloud ``X``, normalized by the number of points and dimensions.
        The updated variance is used to evaluate convergence in the EM algorithm.

        The update follows Equation 23 of the CPD paper (Myronenko & Song, 2010) and accounts for the alignment
        error between the transformed source points and the target points.

        Raises
        ------
        RuntimeError
            If the computed variance becomes non‑positive and cannot be rescued by the fallback tolerance.

        Notes
        -----
        When the newly computed variance is non‑positive, it is replaced by ``self.tolerance / 10`` to
        avoid numerical breakdowns.

        The attribute ``self.sigma2`` is updated in place; ``self.err`` stores the absolute change from
        the previous iteration.

        See Also
        --------
        update_transform : Re‑estimate the deformation matrix before recomputing ``sigma2``.

        References
        ----------
        Myronenko A. and Song X. (2010) **Point set registration: Coherent Point drift**.
        _IEEE Transactions on Pattern Analysis and Machine Intelligence_. 32 (2): 2262-2275.
        DOI: [10.1109/TPAMI.2010.46](https://doi.org/10.1109/TPAMI.2010.46)
        arXiv [PDF](https://arxiv.org/pdf/0905.2635).
        """
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

    def get_registration_parameters(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Return the current Gaussian kernel and deformation matrices.

        Returns
        -------
        tuple of (numpy.ndarray, numpy.ndarray)
            ``G``: Gaussian kernel matrix of shape ``(M, M)``.
            ``W``: Deformable transformation matrix of shape ``(M, D)``.

        Notes
        -----
        The returned objects are *views* of the internal state; modifying them will
        affect the registration instance.
        """
        return self.G, self.W
