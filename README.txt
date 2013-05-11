Quantitative Image Restoration (QIR)
------------------------------------

The code in this directory implements a quantitative image restoration
algorithm for MODIS 500m resolution band 6.

Dependencies:
	Python - http://python.org/
	Numpy - http://numpy.scipy.org/
	Pyhdf - http://pysclint.sourceforge.net/pyhdf/

The code has been tested under at least this configuration:
	x86-64 system
	Python 2.7.2
	Numpy 1.6.1
	Pyhdf 0.8.3
	Scipy 0.12.0

List of files:
	run_qir.py - main executable script
		(run it with -h flag for usage help)
	qir.py - routines implementing QIR
	modis.py - MODIS Level1B reader/writer
	ioutils.py - misc. routines used by modis.py
	utils.py - misc. routines
