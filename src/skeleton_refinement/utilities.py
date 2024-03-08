#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np


def initialize_sigma2(X, Y):
    """Initialize the standard deviation.

    Parameters
    ----------
    X : numpy.ndarray
        ???
    Y : numpy.ndarray
        ???

    Returns
    -------
    numpy.ndarray
        The standard deviation value.
    """
    (N, D) = X.shape
    (M, _) = Y.shape
    XX = np.reshape(X, (1, N, D))
    YY = np.reshape(Y, (M, 1, D))
    XX = np.tile(XX, (M, 1, 1))
    YY = np.tile(YY, (1, N, 1))
    diff = XX - YY
    err = np.multiply(diff, diff)
    return np.sum(err) / (D * M * N)


def gaussian_kernel(Y, beta):
    """Gaussian kernel.

    Parameters
    ----------
    Y : numpy.ndarray
        ???
    beta : float
        ???

    Returns
    -------
    numpy.ndarray
        ???.
    """
    (M, D) = Y.shape
    XX = np.reshape(Y, (1, M, D))
    YY = np.reshape(Y, (M, 1, D))
    XX = np.tile(XX, (M, 1, 1))
    YY = np.tile(YY, (1, M, 1))
    diff = XX - YY
    diff = np.multiply(diff, diff)
    diff = np.sum(diff, 2)
    return np.exp(-diff / (2 * beta))
