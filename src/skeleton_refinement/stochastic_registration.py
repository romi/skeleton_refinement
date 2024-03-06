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


def perform_registration(X, Y):
    """Performs the skeleton optimization using stochastic deformation registration.

    Parameters
    ----------
    X : numpy.ndarray
        The input reference point cloud coordinates of shape `(n_points, dim)`, XYZ sorted.
    Y : numpy.ndarray
        The input reference skeleton coordinates of shape `(n_points, dim)`, XYZ sorted.

    Returns
    -------
    numpy.ndarray
        The transformed skeleton coordinates of shape `(n_points, 3)`, XYZ sorted.
    """
    reg = DeformableRegistration(**{'X': X, 'Y': Y})
    reg.transform_point_cloud()
    if reg.sigma2 is None:
        reg.sigma2 = initialize_sigma2(reg.X, reg.TY)
        reg.q = -reg.err - reg.N * reg.D / 2 * np.log(reg.sigma2)
        while reg.iteration < reg.max_iterations and reg.err > reg.tolerance:
            reg.iterate()
    return reg.TY
