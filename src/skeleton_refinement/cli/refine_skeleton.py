#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
## Skeleton Refinement Tool

A command-line utility for refining plant skeletons by aligning coarse skeleton structures with 3D point cloud data
using a stochastic optimization framework. This tool helps improve the quality of plant representation by pushing
skeleton points toward their proper positions in the original point cloud.

There are two parameters that control the quality of alignment, alpha and beta.

### Usage Examples

```shell
# Basic usage with default parameters
$ refine_skeleton path/to/pointcloud.ply path/to/skeleton.json refined_skeleton.json

# Advanced usage with custom parameters
$ refine_skeleton path/to/pointcloud.ply path/to/skeleton.json refined_skeleton.json --alpha 0.5 --beta 0.1 --knn_mst --n_nei 8
```

### Author

Ayan Chaudhury,
Inria team MOSAIC,
Laboratoire Reproduction et Développement des Plantes,
Univ. Lyon, ENS de Lyon, UCB Lyon 1, CNRS, INRA, Inria
France
"""

import argparse
from pathlib import Path

import numpy as np

from skeleton_refinement.deformable_registration import ALPHA
from skeleton_refinement.deformable_registration import BETA
from skeleton_refinement.expectation_maximization_registration import MAX_ITER
from skeleton_refinement.expectation_maximization_registration import TOL
from skeleton_refinement.io import load_json
from skeleton_refinement.io import load_nx
from skeleton_refinement.io import load_ply
from skeleton_refinement.io import load_xyz
from skeleton_refinement.io import save_json
from skeleton_refinement.io import save_nx
from skeleton_refinement.stochastic_registration import knn_mst
from skeleton_refinement.stochastic_registration import perform_registration

URL = "https://romi.github.io/skeleton_refinement/"
IN_FMT = ['json', 'p', 'ply', 'txt', 'xyz']
OUT_FMT = ['json', 'p', 'txt', 'xyz']


def parsing():
    parser = argparse.ArgumentParser(
        description='Plant skeleton optimization using stochastic framework on point cloud data.',
        epilog=f"See online documentation for more details: {URL}")
    parser.add_argument(
        'pcd', type=Path,
        help="Path to the point cloud used to build the skeleton. "
             f"Allowed extensions are in {IN_FMT}.")
    parser.add_argument(
        'skeleton', type=Path,
        help="Path to the skeleton to refine. "
             f"Allowed extensions are in {IN_FMT}.")
    parser.add_argument(
        'out', type=Path,
        help="Path to use to save the refined skeleton."
             f"Allowed extensions are in {OUT_FMT}. "
             "JSON format export nodes coordinates under 'points' and edges as pairs of node indexes under 'lines'. "
             "Pickle format (.p) will save the resulting skeleton as networkx graph. "
             "Text formats (.txt or .xyz) will save the skeleton points as space delimited text XYZ coordinates. "
    )

    em_opt = parser.add_argument_group("EM algorithm options")
    em_opt.add_argument(
        '--alpha', type=float, default=ALPHA,
        help="Alpha value. "
             f"Default is {ALPHA}.")
    em_opt.add_argument(
        '--beta', type=float, default=BETA,
        help="Beta value. "
             f"Default is {BETA}.")
    em_opt.add_argument(
        '--max_iter', type=int, default=MAX_ITER,
        help="Maximum number of iterations of the EM algorithm to perform. "
             f"Default is '{MAX_ITER}'.")
    em_opt.add_argument(
        '--tol', type=float, default=TOL,
        help="Tolerance to use to stop the iterations of the EM algorithm. "
             f"Default is '{TOL}'.")

    tree_opt = parser.add_argument_group("tree options")
    tree_opt.add_argument(
        '--knn_mst', action="store_true",
        help="Update the tree structure with minimum spanning tree on knn-graph.")
    tree_opt.add_argument(
        '--n_nei', type=int, default=5,
        help="The number of neighbors to search for in `skeleton_points`. "
             "Default is '5'.")
    tree_opt.add_argument(
        '--knn_algo', type=str, default="kd_tree",
        choices=['auto', 'ball_tree', 'kd_tree', 'brute'],
        help="The algorithm to use for computing the kNN distance. "
             "Default is 'kd_tree'.")
    tree_opt.add_argument(
        '--mst_algo', type=str, default="kruskal",
        choices=['kruskal', 'prim', 'boruvka'],
        help="The algorithm to use for computing the minimum spanning tree. "
             "Default is 'kruskal'.")

    return parser


def file_loader(fname):
    """Load point cloud or skeleton from a file."""
    if fname.suffix in [".xyz", ".txt"]:
        xyz = load_xyz(fname)
    elif fname.suffix == ".json":
        xyz = load_json(fname, "points")
    elif fname.suffix == ".ply":
        xyz = load_ply(fname)
    elif fname.suffix == ".p":
        xyz = load_nx(fname)
    else:
        raise IOError(f"Unknown input file format '{fname.suffix}' for file '{fname}'! Choose from {IN_FMT}.")
    return xyz


def file_writer(fname, skel):
    """Write skeleton to a file."""
    if fname.suffix in [".xyz", ".txt"]:
        np.savetxt(fname, skel, delimiter=' ')
    elif fname.suffix == ".json":
        save_json(fname, skel)
    elif fname.suffix == ".p":
        save_nx(fname, skel)
    else:
        raise IOError(f"Unknown output file format {fname.suffix}!")
    return


def main():
    # - Parse the input arguments to variables:
    parser = parsing()
    args = parser.parse_args()
    # - Check the ouptut file format:
    try:
        assert args.out.suffix in OUT_FMT
    except AssertionError:
        raise IOError(f"Unknown output file format {args.out.suffix}, choose from {OUT_FMT}.")

    # - Load the point cloud data from the file:
    pcd = file_loader(args.pcd)
    # - Load the skeleton data from the file:
    skel = file_loader(args.skeleton)
    # - Perform stochastic optimization
    refined_skel = perform_registration(pcd, skel, alpha=args.alpha, beta=args.beta,
                                        max_iterations=args.max_iter, tolerance=args.tol)
    # - Update skeleton structure if required
    if args.knn_mst:
        refined_skel = knn_mst(refined_skel, n_neighbors=args.n_nei,
                               knn_algorithm=args.knn_algo, mst_algorithm=args.mst_algo)
    # - Save the refined skeleton:
    file_writer(args.out, refined_skel)


if __name__ == "__main__":
    main()
