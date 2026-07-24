# Changelog for v0.1.3

## Features
- Added PyVista visualization utilities for pointcloud and skeleton
- Added `skeleton()` helper to load full skeleton data
- Added data module with helpers for bundled pointcloud and skeleton
- Added real-plant pointcloud and skeleton dataset tracked with Git LFS
- Added LFS tracking for skeleton refinement data files
- Added Git LFS & package data instructions to README.md
- Added ipython to conda environment setup in README
- Added setuptools configuration for package data

## Documentation
- Updated mkdocs configuration for Python handler
- Refine documentation headings and add type hints
- Improve documentation and clarify initialization in registration modules
- Update stochastic registration examples to use data helpers and clarify documentation
- Improve `load_json` handling and documentation
- Update documentation wording and style in docs/index.md
- Improve CLI documentation and argument help for skeleton refinement tool
- Reformat IO module docstrings
- Improve README wording and consistency
- Add arXiv reference and pycpd links to README and capitalize Python
- Add new ROMI logo to README and assets

## Dependencies
- Added pyvista dependency
- Added tqdm dependency (previously added in commit 712657a2eae3df10f3355cb807545dcac795649d)
- Corrected psutil typo
- Removed unused GUI extra
- Added `psutils` to core dependencies and introduced `gui` optional extra

## Bug Fixes
- Fixed README typos
- Simplified license entry and fixed package-data path
- Simplified argument handling and cleaned up docstrings in stochastic registration
- Improved documentation and clarified EM registration API
- Refine DeformableRegistration documentation and add automatic sigma2 initialization

## Packaging
- Bump package version to 0.1.3
- Updated package data configuration
- Enhanced IO module with type hints, clearer returns, and expanded documentation