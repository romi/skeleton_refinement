#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This program performs refinement of an existing skeleton (in the form of 3D point cloud data)
of a plant starting from it's initial "coarse" structure. The coarse skeleton is "rectified"
via a stochastic optimization framework, by "pushing" the skeleton points towards the original
point cloud data so that they get "aligned" in space. There are two parameters that control
the quality of alignment, alpha and beta (denoted as 'myAlpha' and 'myBeta' in 'param_settings.py')

Author: Ayan Chaudhury
INRIA team MOSAIC

"""
import argparse
from pathlib import Path

import numpy as np

from skeleton_refinement.stochastic_registration import perform_registration
from skeleton_refinement.utilities import load_xyz, load_json, load_ply


def parsing():
    parser = argparse.ArgumentParser(description='Refine skeleton.')
    parser.add_argument('pcd', type=str,
                        help='Path to the point cloud used to build the skeleton.')
    parser.add_argument('skeleton', type=str,
                        help='Path to the skeleton to refine.')
    parser.add_argument('out', type=str,
                        help='Path to use to save the refined skeleton.')
    return parser


def file_loader(fname):
    """Load point cloud or skeleton from a file."""
    fname = Path(fname)
    if fname.suffix == ".xyz":
        xyz = load_xyz(fname)
    elif fname.suffix == ".json":
        xyz = load_json(fname, "points")
    elif fname.suffix == ".ply":
        xyz = load_ply(fname)
    else:
        raise IOError("Unknown file format!")
    return xyz


def main():
    # - Parse the input arguments to variables:
    parser = parsing()
    args = parser.parse_args()

    # pointCloudFileName = 'plantPC_cleaned.xyz'
    # skeletonFileName = 'mtgSkeletonPoints.xyz'
    # transformedPointCloudFileName = 'refinedSkeleton.xyz'

    X_original_PC = file_loader(args.pcd)
    Y_skeleton_PC = file_loader(args.skeleton)
    # Perform stochastic optimization
    reg_x, reg_y = perform_registration(X_original_PC, Y_skeleton_PC)
    # Save the refined skeleton:
    np.savetxt(args.out, reg_y, delimiter=' ')


if __name__ == "__main__":
    main()
