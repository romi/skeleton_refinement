#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This code implements the basic structure of for performing the stochastic
optimization algorithm. Given two set of discrete points, this code returns
the transformed point set.
"""

import numpy as np

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
