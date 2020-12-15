StatsUtils.py [![Unlicensed work](https://raw.githubusercontent.com/unlicense/unlicense.org/master/static/favicon.png)](https://unlicense.org/)
=============
~~[wheel (GitLab)](https://gitlab.com/KOLANICH-libs/StatsUtils.py/-/jobs/artifacts/master/raw/dist/StatsUtils-0.CI-py3-none-any.whl?job=build)~~
[wheel (GHA via `nightly.link`)](https://nightly.link/KOLANICH-libs/StatsUtils.py/workflows/CI/master/StatsUtils-0.CI-py3-none-any.whl)
~~![GitLab Build Status](https://gitlab.com/KOLANICH-libs/StatsUtils.py/badges/master/pipeline.svg)~~
~~![GitLab Coverage](https://gitlab.com/KOLANICH-libs/StatsUtils.py/badges/master/coverage.svg)~~
~~[![GitHub Actions](https://github.com/KOLANICH-libs/StatsUtils.py/workflows/CI/badge.svg)](https://github.com/KOLANICH-libs/StatsUtils.py/actions/)~~
[![Libraries.io Status](https://img.shields.io/librariesio/github/KOLANICH-libs/StatsUtils.py.svg)](https://libraries.io/github/KOLANICH-libs/StatsUtils.py)
[![Code style: antiflash](https://img.shields.io/badge/code%20style-antiflash-FFF.svg)](https://codeberg.org/KOLANICH-tools/antiflash.py)

These is a temporary package with some routines that I think must be a part of `scipy.stats` and `numpy`, but curently they are not. The end goal is to upsteam as much as possible of this package into `scipy` and get rid of it.

It monkey-patches `scipy.stats.norm` and `scipy.stats.truncnorm`, adds `rv_empirical` and adds some useful methods for adaptive meshes for histograms.

The method uses modulus of PDF derivative to compute a mesh and can be used for any function in fact.
1. The function is differentiated.
2. The modulus of derivative is computed.
3. It is integrated and normalized to become a CDF and the CDF is transformed into PPF via interpolation.
4. A mesh is computed by transforming a uniform mesh into non-uniform via PPF.

Read [the tutorial](./tutorial.ipynb)[![NBViewer](https://nbviewer.org/static/ico/ipynb_icon_16x16.png)](https://nbviewer.org/urls/codeberg.org/KOLANICH-libs/StatsUtils.py/raw/branch/master/tutorial.ipynb).
