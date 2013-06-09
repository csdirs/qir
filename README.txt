Quantitative Image Restoration (QIR)
====================================

The code in this directory implements a quantitative image restoration
algorithm[1] for MODIS 500m resolution band 6.

Dependencies
------------
	Python - http://python.org/
	Numpy - http://numpy.scipy.org/
	Scipy - http://www.scipy.org/
	Pyhdf - http://pysclint.sourceforge.net/pyhdf/
	argparse (for Python < 2.7) - https://pypi.python.org/pypi/argparse

The code has been tested under at least this configuration:
	x86-64 system
	Python 2.7.2
	Numpy 1.6.1
	Scipy 0.12.0
	Pyhdf 0.8.3

Installing
----------
To install, run:

	python2 setup.py install

A 'qir' command will be installed. Run it with -h flag for usage help.


References
----------
1. Gladkova, I., Grossberg, M. D., Shahriar, F., Bonev, G., & Romanov, P.
(2012).  Quantitative restoration for MODIS band 6 on Aqua. Geoscience and
Remote Sensing, IEEE Transactions on, 50(6), 2409-2416.
