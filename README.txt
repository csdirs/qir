Quantitative Image Restoration (QIR)
====================================

The code in this directory implements a quantitative image restoration
algorithm [QIR2012]_ for MODIS 500m resolution band 6.

Dependencies
------------

QIR requires the following dependencies:

	- Python - http://python.org/
	- Numpy - http://numpy.scipy.org/
	- Scipy - http://www.scipy.org/
	- Pyhdf - http://pysclint.sourceforge.net/pyhdf/
	- argparse (for Python < 2.7) - https://pypi.python.org/pypi/argparse

Installing
----------

To install, run::

	python2 setup.py install

A ``qir`` command will be installed. Run it with -h flag for usage help.

References
----------

.. [QIR2012] Gladkova, I., Grossberg, M. D., Shahriar, F., Bonev, G., &
	Romanov, P.  (2012).  Quantitative restoration for MODIS band 6 on Aqua.
	Geoscience and Remote Sensing, IEEE Transactions on, 50(6), 2409-2416.
