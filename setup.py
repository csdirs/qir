#!/usr/bin/env python2

from distutils.core import setup

setup(name="qir",
        version="1.0",
        description="Quatitative image restoration for MODIS 500m resolution band 6.",
        maintainer="Fazlul Shahriar",
        maintainer_email="fshahriar@gc.cuny.edu",
        license="New BSD",
        url="https://bitbucket.org/fhs1/qir/overview",
        packages=["qir"],
        scripts=["scripts/qir"],
        requires=["numpy", "pyhdf", "scipy"],
)
