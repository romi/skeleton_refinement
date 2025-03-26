# Skeleton Refinement

[![Anaconda-Server Badge](https://anaconda.org/romi-eu/skeleton_refinement/badges/version.svg)](https://anaconda.org/romi-eu/skeleton_refinement)
[![Anaconda-Server Badge](https://anaconda.org/romi-eu/skeleton_refinement/badges/platforms.svg)](https://anaconda.org/romi-eu/skeleton_refinement)
[![Anaconda-Server Badge](https://anaconda.org/romi-eu/skeleton_refinement/badges/license.svg)](https://anaconda.org/romi-eu/skeleton_refinement)

The documentation of the _Plant Imager_ project can be found here: https://docs.romi-project.eu/plant_imager/

The API documentation of the `skeleton_refinement` library can be found here: https://romi.github.io/skeleton_refinement/ 

## About

This library is intended to provide the implementation of a skeleton refinement method published here:

Chaudhury A. and Godin C. (2020) **Skeletonization of Plant Point Cloud Data Using Stochastic Optimization Framework**.
_Front. Plant Sci._ 11:773.
DOI: [10.3389/fpls.2020.00773](https://doi.org/10.3389/fpls.2020.00773).

![Skeleton refinement result on arabidopsis data.](docs/assets/images/arabidopsis_example.png)

This is a part of the implementation of the stochastic registration algorithm based on the following paper:
Myronenko A. and Song X. (2010) **Point set registration: Coherent Point drift**.
_IEEE Transactions on Pattern Analysis and Machine Intelligence_. 32 (2): 2262-2275.
DOI: [10.1109/TPAMI.2010.46](https://doi.org/10.1109/TPAMI.2010.46)


## Installation

We strongly advise to create isolated environments to install the ROMI libraries.

We often use `conda` as an environment and python package manager.
If you do not yet have `miniconda3` installed on your system, have a look [here](https://docs.conda.io/en/latest/miniconda.html).

The `skeleton_refinement` package is available from the `romi-eu` channel.

### Existing conda environment
To install the `skeleton_refinement` conda package in an existing environment, first activate it, then proceed as follows:
```shell
conda install skeleton_refinement -c romi-eu
```

### New conda environment
To install the `skeleton_refinement` conda package in a new environment, here named `romi`, proceed as follows:
```shell
conda create -n romi skeleton_refinement -c romi-eu
```

### Installation from sources
To install this library, simply clone the repo and use `pip` to install it and the required dependencies.
Again, we strongly advise to create a `conda` environment.

All this can be done as follows:
```shell
git clone https://github.com/romi/skeleton_refinement.git
cd skeleton_refinement
conda create -n skeleton_refinement 'python =3.10'
conda activate skeleton_refinement  # do not forget to activate your environment!
python -m pip install -e .  # install the sources
```

Note that the `-e` option is to install the `skeleton_refinement` sources in "developer mode".
That is, if you make changes to the source code of `skeleton_refinement` you will not have to `pip install` it again.


## Usage

### Example dataset

First, we download an example dataset from Zenodo, named `real_plant_analyzed`, to play with:

```shell
wget https://zenodo.org/records/10379172/files/real_plant_analyzed.zip
unzip real_plant_analyzed.zip -d /tmp
```

It contains:
  * a plant point cloud under `PointCloud_1_0_1_0_10_0_7ee836e5a9/PointCloud.ply`
  * a plant skeleton under `CurveSkeleton__TriangleMesh_0393cb5708/CurveSkeleton.json`
  * a plant tree graph under `TreeGraph__False_CurveSkeleton_c304a2cc71/TreeGraph.p`


### CLI

You may use the `refine_skeleton` CLI to refine a given skeleton using the original point cloud: 

```shell
export DATA_PATH="/tmp/real_plant_analyzed"
refine_skeleton \
  ${DATA_PATH}/PointCloud_1_0_1_0_10_0_7ee836e5a9/PointCloud.ply \
  ${DATA_PATH}/CurveSkeleton__TriangleMesh_0393cb5708/CurveSkeleton.json \
  ${DATA_PATH}/optimized_skeleton.txt
```

### Python API

Here is a minimal example how to use the `skeleton_refinement` library in Python:

```python
from skeleton_refinement.stochastic_registration import perform_registration
from skeleton_refinement.io import load_json, load_ply

pcd = load_ply("/tmp/real_plant_analyzed/PointCloud_1_0_1_0_10_0_7ee836e5a9/PointCloud.ply")
skel = load_json("/tmp/real_plant_analyzed/CurveSkeleton__TriangleMesh_0393cb5708/CurveSkeleton.json", "points")
# Perform stochastic optimization
refined_skel = perform_registration(pcd, skel)

import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(*pcd.T, marker='.', color='black')
ax.scatter(*skel.T, marker='o', color='r')
ax.scatter(*refined_skel.T, marker='o', color='b')
ax.set_aspect('equal')
plt.show()
```

A detailed documentation of the Python API is available here: https://romi.github.io/skeleton_refinement/reference.html


## Developers & contributors

### Unitary tests

Some tests are defined in the `tests` directory.
We use `nose2` to call them as follows:

```shell
nose2 -v -C
```

Notes:

- the configuratio[mkdocs.yml](mkdocs.yml)n file used by `nose2` is `unittests.cfg`
- the `-C` option generate a coverage report, as defined by the `.coveragerc` file.
- this requires the `nose2` & `coverage` packages listed in the `requirements.txt` file.

You first have to install the library from sources as explained [here](#installation-from-sources).

### Conda packaging
Start by installing the required `conda-build` & `anaconda-client` conda packages in the `base` environment as follows:
```shell
conda install -n base conda-build anaconda-client
```

#### Build a conda package
To build the `romitask` conda package, from the root directory of the repository and the `base` conda environment, run:
```shell
conda build conda/recipe/ -c conda-forge --user romi-eu
```

If you are struggling with some of the modifications you made to the recipe, 
notably when using environment variables or Jinja2 stuffs, you can always render the recipe with:
```shell
conda render conda/recipe/
```

The official documentation for `conda-render` can be found [here](https://docs.conda.io/projects/conda-build/en/stable/resources/commands/conda-render.html).

#### Upload a conda package
To upload the built package, you need a valid account (here `romi-eu`) on [anaconda.org](www.anaconda.org) & to log ONCE
with `anaconda login`, then:
```shell
anaconda upload ~/miniconda3/conda-bld/linux-64/skeleton_refinement*.tar.bz2 --user romi-eu
```

#### Clean builds
To clean the source and build intermediates:
```shell
conda build purge
```

To clean **ALL** the built packages & build environments:
```shell
conda build purge-all
```

