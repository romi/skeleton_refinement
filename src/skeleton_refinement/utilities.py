#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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
        The XYZ coordinates of the point cloud or skeleton.
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
    """Load a point cloud coordinates.

    Parameters
    ----------
    filename : str or pathlib.Path
        Path to the point cloud ofile to parse.

    Returns
    -------
    numpy.ndarray
        The XYZ coordinates of the point cloud or skeleton.
    """
    from plyfile import PlyData
    plydata = PlyData.read(filename)
    X = np.array([plydata['vertex']['x'], plydata['vertex']['y'], plydata['vertex']['z']]).T
    return X

def load_json(filename, key=None):
    """Load a point cloud or skeleton file from a json file.
    Parameters
    ----------
    filename : str or pathlib.Path
        Path to the point cloud or skeleton file to parse.
    key : str, optional
        The key of the JSON dictionary containing the point cloud or skeleton coordinates to load.

    Returns
    -------
    numpy.ndarray
        The XYZ coordinates of the point cloud or skeleton.
    """
    import json
    with open(filename, mode='rb') as f:
        X = json.load(f)

    if key is not None:
        X = X[key]
    return np.array(X)


def initialize_sigma2(X, Y):
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
    (M, D) = Y.shape
    XX = np.reshape(Y, (1, M, D))
    YY = np.reshape(Y, (M, 1, D))
    XX = np.tile(XX, (M, 1, 1))
    YY = np.tile(YY, (1, M, 1))
    diff = XX - YY
    diff = np.multiply(diff, diff)
    diff = np.sum(diff, 2)
    return np.exp(-diff / (2 * beta))
