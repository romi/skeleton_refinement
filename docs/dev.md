# Developers & contributors

## Unitary tests

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

## Conda packaging
Start by installing the required `conda-build` & `anaconda-client` conda packages in the `base` environment as follows:
```shell
conda install -n base conda-build anaconda-client
```

### Build a conda package
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

### Upload a conda package
To upload the built package, you need a valid account (here `romi-eu`) on [anaconda.org](www.anaconda.org) & to log ONCE
with `anaconda login`, then:
```shell
anaconda upload ~/miniconda3/conda-bld/linux-64/skeleton_refinement*.tar.bz2 --user romi-eu
```

### Clean builds
To clean the source and build intermediates:
```shell
conda build purge
```

To clean **ALL** the built packages & build environments:
```shell
conda build purge-all
```

