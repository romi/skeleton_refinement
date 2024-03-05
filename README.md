# Skeleton Refinement

[![Anaconda-Server Badge](https://anaconda.org/romi-eu/skeleton_refinement/badges/version.svg)](https://anaconda.org/romi-eu/skeleton_refinement)
[![Anaconda-Server Badge](https://anaconda.org/romi-eu/skeleton_refinement/badges/platforms.svg)](https://anaconda.org/romi-eu/skeleton_refinement)
[![Anaconda-Server Badge](https://anaconda.org/romi-eu/skeleton_refinement/badges/license.svg)](https://anaconda.org/romi-eu/skeleton_refinement)

The documentation of the _Plant Imager_ project can be found here: https://docs.romi-project.eu/plant_imager/

The API documentation of the `skeleton_refinement` library can be found here: https://romi.github.io/skeleton_refinement/ 

## About

This library is intended to provide the implementation of a skeleton refinement method published here:

Chaudhury A. and Godin C. (2020) **Skeletonization of Plant Point Cloud Data Using Stochastic Optimization Framework**. _Front. Plant Sci._ 11:773. doi: [10.3389/fpls.2020.00773](https://doi.org/10.3389/fpls.2020.00773).


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

### Tests the library
First you need to install the tests tools:
```shell
python -m pip install -e .[test]
```

Then, to test the `skeleton_refinement` library:
 - Run all tests with verbose output (from the `skeleton_refinement` root directory):
    ```shell
    nose2 -s tests/ -v
    ```
 - Run all tests with coverage report (from the `skeleton_refinement` root directory):
    ```shell
    nose2 -s tests/ --with-coverage
    ```


## Usage

### Example dataset
First download an example dataset to play with:
```shell
cd tmp/
wget https://zenodo.org/records/10379172/files/real_plant_analyzed.zip

unzip real_plant_analyzed.zip
```

### CLI
```shell
refine_skeleton \
  real_plant_analyzed/PointCloud_1_0_1_0_10_0_7ee836e5a9/PointCloud.ply \
  real_plant_analyzed/CurveSkeleton__TriangleMesh_0393cb5708/CurveSkeleton.json \
  real_plant_analyzed/optimized_skeleton.txt
```

### Python API

Here is a minimal example how to use the `skeleton_refinement` library in Python:

```python
from skeleton_refinement.stochastic_registration import perform_registration
from skeleton_refinement.utilities import load_json, load_ply

X_original_PC = load_ply("real_plant_analyzed/PointCloud_1_0_1_0_10_0_7ee836e5a9/PointCloud.ply")
Y_skeleton_PC = load_json("real_plant_analyzed/CurveSkeleton__TriangleMesh_0393cb5708/CurveSkeleton.json", "points")
# Perform stochastic optimization
reg_x, reg_y = perform_registration(X_original_PC, Y_skeleton_PC)

import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(*reg_x.T, marker='.', color='black')
ax.scatter(*Y_skeleton_PC.T, marker='o', color='r')
ax.scatter(*reg_y.T, marker='o', color='b')
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

- the configuration file used by `nose2` is `unittests.cfg`
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

