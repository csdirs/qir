#!/usr/bin/env python2

from distutils.core import setup

version = "1.2"

with open("qir/_version.py", "w") as f:
    f.write("__version__ = \"%s\"\n" % (version,))

setup(name="qir",
        version=version,
        description="Quatitative image restoration for MODIS 500m resolution band 6.",
        maintainer="Fazlul Shahriar",
        maintainer_email="fshahriar@gc.cuny.edu",
        license="New BSD",
        url="https://bitbucket.org/fhs1/qir/overview",
        packages=["qir"],
        scripts=["scripts/qir", "scripts/qir1km"],
        requires=["numpy", "pyhdf", "scipy"],
)
