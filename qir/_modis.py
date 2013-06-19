"""
Read and write MODIS data sets (:mod:`modis`)
=================================================================

MODIS (or Moderate Resolution Imaging Spectroradiometer) is an instrument
aboard the Terra (EOS AM) and Aqua (EOS PM) satellites.
(See: http://modis.gsfc.nasa.gov/)

.. autosummary::

        destripe
        Level1B
        Level1BBand
        Level1BVariable

"""

import numpy as np

from . import _ioutils as ioutils
from . import _utils as utils

_QKM_BAND_NAMES = '1,2'.split(',')
_HKM_BAND_NAMES = '3,4,5,6,7'.split(',')
_1KM_REF_BAND_NAMES = '8,9,10,11,12,13lo,13hi,14lo,14hi,15,16,17,18,19,26'.split(',')
_1KM_EMISSIVE_BAND_NAMES = '20,21,22,23,24,25,27,28,29,30,31,32,33,34,35,36'.split(',')
_SUPPORTED_PARAMS = ['raw', 'radiance', 'reflectance', 'corrected_counts', 'temperature']

_BAND_VARNAMES = {
    (1000, '1')     : ('EV_250_Aggr1km_RefSB', _QKM_BAND_NAMES),
    (1000, '2')     : ('EV_250_Aggr1km_RefSB', _QKM_BAND_NAMES),
    (1000, '3')     : ('EV_500_Aggr1km_RefSB', _HKM_BAND_NAMES),
    (1000, '4')     : ('EV_500_Aggr1km_RefSB', _HKM_BAND_NAMES),
    (1000, '5')     : ('EV_500_Aggr1km_RefSB', _HKM_BAND_NAMES),
    (1000, '6')     : ('EV_500_Aggr1km_RefSB', _HKM_BAND_NAMES),
    (1000, '7')     : ('EV_500_Aggr1km_RefSB', _HKM_BAND_NAMES),
    (1000, '8')     : ('EV_1KM_RefSB', _1KM_REF_BAND_NAMES),
    (1000, '9')     : ('EV_1KM_RefSB', _1KM_REF_BAND_NAMES),
    (1000, '10')    : ('EV_1KM_RefSB', _1KM_REF_BAND_NAMES),
    (1000, '11')    : ('EV_1KM_RefSB', _1KM_REF_BAND_NAMES),
    (1000, '12')    : ('EV_1KM_RefSB', _1KM_REF_BAND_NAMES),
    (1000, '13lo')  : ('EV_1KM_RefSB', _1KM_REF_BAND_NAMES),
    (1000, '13hi')  : ('EV_1KM_RefSB', _1KM_REF_BAND_NAMES),
    (1000, '14lo')  : ('EV_1KM_RefSB', _1KM_REF_BAND_NAMES),
    (1000, '14hi')  : ('EV_1KM_RefSB', _1KM_REF_BAND_NAMES),
    (1000, '15')    : ('EV_1KM_RefSB', _1KM_REF_BAND_NAMES),
    (1000, '16')    : ('EV_1KM_RefSB', _1KM_REF_BAND_NAMES),
    (1000, '17')    : ('EV_1KM_RefSB', _1KM_REF_BAND_NAMES),
    (1000, '18')    : ('EV_1KM_RefSB', _1KM_REF_BAND_NAMES),
    (1000, '19')    : ('EV_1KM_RefSB', _1KM_REF_BAND_NAMES),
    (1000, '20')    : ('EV_1KM_Emissive', _1KM_EMISSIVE_BAND_NAMES),
    (1000, '21')    : ('EV_1KM_Emissive', _1KM_EMISSIVE_BAND_NAMES),
    (1000, '22')    : ('EV_1KM_Emissive', _1KM_EMISSIVE_BAND_NAMES),
    (1000, '23')    : ('EV_1KM_Emissive', _1KM_EMISSIVE_BAND_NAMES),
    (1000, '24')    : ('EV_1KM_Emissive', _1KM_EMISSIVE_BAND_NAMES),
    (1000, '25')    : ('EV_1KM_Emissive', _1KM_EMISSIVE_BAND_NAMES),
    (1000, '26')    : ('EV_Band26', None),
    (1000, '27')    : ('EV_1KM_Emissive', _1KM_EMISSIVE_BAND_NAMES),
    (1000, '28')    : ('EV_1KM_Emissive', _1KM_EMISSIVE_BAND_NAMES),
    (1000, '29')    : ('EV_1KM_Emissive', _1KM_EMISSIVE_BAND_NAMES),
    (1000, '30')    : ('EV_1KM_Emissive', _1KM_EMISSIVE_BAND_NAMES),
    (1000, '31')    : ('EV_1KM_Emissive', _1KM_EMISSIVE_BAND_NAMES),
    (1000, '32')    : ('EV_1KM_Emissive', _1KM_EMISSIVE_BAND_NAMES),
    (1000, '33')    : ('EV_1KM_Emissive', _1KM_EMISSIVE_BAND_NAMES),
    (1000, '34')    : ('EV_1KM_Emissive', _1KM_EMISSIVE_BAND_NAMES),
    (1000, '35')    : ('EV_1KM_Emissive', _1KM_EMISSIVE_BAND_NAMES),
    (1000, '36')    : ('EV_1KM_Emissive', _1KM_EMISSIVE_BAND_NAMES),
    (500, '1')      : ('EV_250_Aggr500_RefSB', _QKM_BAND_NAMES),
    (500, '2')      : ('EV_250_Aggr500_RefSB', _QKM_BAND_NAMES),
    (500, '3')      : ('EV_500_RefSB', _HKM_BAND_NAMES),
    (500, '4')      : ('EV_500_RefSB', _HKM_BAND_NAMES),
    (500, '5')      : ('EV_500_RefSB', _HKM_BAND_NAMES),
    (500, '6')      : ('EV_500_RefSB', _HKM_BAND_NAMES),
    (500, '7')      : ('EV_500_RefSB', _HKM_BAND_NAMES),
    (250, '1')      : ('EV_250_RefSB', _QKM_BAND_NAMES),
    (250, '2')      : ('EV_250_RefSB', _QKM_BAND_NAMES),
}


def destripe(img, resolution, valid_range, nbins=100, skipdet=None):
    """Destripe a modis band image using histogram matching.

    See also :func:`Level1BBand.destripe`, which should be used
    instead whenever possible.

    Parameters
    ----------
    img : 2d ndarray
        Image to be destriped.
    resolution : str
        Resolution of image in meters: 1000, 500, or 250.
    valid_range : tuple
        Minimum and maximum valid value for the band.
    nbins : int, optional
        Bin size for the empirical CDF.
    skipdet : list
        List of detectors that shouldn't be destriped.

    Returns
    -------
    new_img : 2d ndarray
        Destriped image.
    """
    res2nsens = {1000:10, 500:20, 250:40}
    if resolution not in res2nsens:
        raise ValueError("invalid resolution %s" % resolution)
    nsens = res2nsens[resolution]

    destriped = img.copy()
    lastrow = (img.shape[0]//nsens)*nsens
    strippedsens = range(nsens)
    if skipdet is not None:
        strippedsens = [s for s in xrange(nsens) if s not in skipdet]

    # use first detector/sensor as reference
    goodsen = strippedsens.pop(0)

    D1 = img[goodsen:lastrow:nsens, :].ravel()
    D1_valid = D1[(valid_range[0] < D1) & (D1 < valid_range[1])]
    m = D1_valid.min()
    M = D1_valid.max()
    H = np.zeros((2, nbins))

    for sen in strippedsens:
        D = img[sen::nsens, :]
        D_shape = D.shape
        D = D.ravel()
        D_valid = D[(valid_range[0] < D) & (D < valid_range[1])]
        x = np.linspace(min(m, np.min(D_valid)), max(M, np.max(D_valid)), nbins+1)
        WH, _ = np.histogram(D1_valid, bins=x, normed=True)
        H[0,:] = np.cumsum(WH/float(len(D1_valid)))
        WH, _ = np.histogram(D_valid, bins=x, normed=True)
        H[1,:] = np.cumsum(WH/float(len(D_valid)))
        y = np.zeros(nbins)
        for i in xrange(nbins):
            indL = np.where(H[0,:] <= H[1,i])[0]
            if len(indL) == 0 or len(indL) == nbins:
                y[i] = x[i]
            else:
                pos = indL.max()
                xL = x[pos]
                fL = H[0, pos]
                xR = x[pos+1]
                fR = H[0, pos+1]
                y[i] = xL + (H[1,i]-fL)*(xR-xL)/float(fR-fL)

        B = np.interp(D_valid, x[:-1], y)
        D[(valid_range[0] < D) & (D < valid_range[1])] = B
        destriped[sen::nsens, :] = D.reshape(D_shape)
    return destriped

class FillInvalidError(Exception):
    """Create an instance of FillInvalidError exception.

    This exception is raised by :func:`Level1BVariable.fill_invalid`.
    """
    pass

class Level1BVariable(object):
    """Create an instance of Level 1B variable.

    The constructor is not meant to be used directly. Use members of
    :class:`Level1B` instead.
    """
    def __init__(self, level1b, vname, index):
        self._level1b = level1b
        self._var = level1b._sd.get_dataset(vname)
        self._index = index
        self._attributes = self._var.getattr()

    def close(self):
        """End access to the band HDF variable.
        """
        return self._var.close()

    @property
    def sds(self):
        """The underlying HDF variable object (pyhdf.SD.SDS).
        """
        return self._var.SDS

    @property
    def level1b(self):
        """The associated :class:`Level1B` instance.
        """
        return self._level1b

    @property
    def shape(self):
        """Shape of the band, a 2-tuple of ints.
        """
        s = self._var.shape
        if len(s) == 2:
            return s
        if len(s) == 3:
            return s[1:]
        raise Exception("unexpected HDF variable shape %s" % s)

    @property
    def attributes(self):
        """HDF variable attributes (dict: str -> object).
        """
        return self._attributes

    def valid_range(self):
        """Returns the valid range of the data.

        Returns
        -------
        vr : ndarray of shape (2,)
            The valid range where the first element is the lower bound, and the
            second the upper bound.
        """
        return np.array(self._attributes['valid_range'])

    def is_invalid(self, data):
        """Returns whether the data is outside the valid range.

        Parameters
        ----------
        data : ndarray
            The data.

        Returns
        -------
        invalid : bool ndarray
            A pixel is True iff it is outside the valid range.
        """
        vr = self.valid_range()
        return (data < vr[0]) | (vr[1] < data)

    def is_valid(self, data):
        """Returns whether the data is inside the valid range.

        Parameters
        ----------
        data : ndarray
            The data.

        Returns
        -------
        valid : bool ndarray
            A pixel is True iff it is inside the valid range.
        """
        return ~self.is_invalid(data)

    def fill_invalid(self, img, winsize=11, maxinvalid=0.35, pad=True):
        """A wrapper around :func:`utils.fillinvalid`.

        The only difference is that the invalid mask argument is not needed.
        """
        if np.ma.isMaskedArray(img) and np.any(img.mask):
            raise FillInvalidError("image contains %d _FillValue pixels" % np.sum(img.mask))
        invalid = self.is_invalid(img)
        return utils.fillinvalid(img, invalid=invalid,
            winsize=winsize, maxinvalid=maxinvalid, pad=pad)

    def read(self, start=None, count=None):
        """Read 2D data from file.

        Parameters
        ----------
        start : 2-tuple, optional
            Start of crop (top left corner).
        count : 2-tuple, optional
            Size of crop.

        Returns
        -------
        data : 2D masked ndarray
            Band data.
        """
        img = self._var.readarray(self._index, start=start, count=count)
        return np.ma.masked_equal(img,
                np.array(self._attributes['_FillValue'], dtype=img.dtype))

    def write(self, img, start=None):
        """Write 2D data to file.

        Parameters
        ----------
        img : 2D ndarray
            Data to be written.
        start : 2-tuple, optional
            Start of crop (top left corner).
        """
        if img.ndim != 2:
            raise ValueError('Image img has dimension %d' % img.ndim)

        # round if it's an unsigned or signed integer type
        typ = self._var.getdtype()
        if typ.kind == 'u' or typ.kind == 'i':
            img = np.round(img).astype(typ)

        if np.ma.isMaskedArray(img):
            img = img.filled(fill_value=self._attributes['_FillValue'])
        return self._var.writearray(self._index, img, start=start)

class Level1BBand(Level1BVariable):
    """Create an instance of Level 1B band.

    This is a subclass of :class:`Level1BVariable`. The constructor is not
    meant to be used directly. Use members of :class:`Level1B` instead.
    """
    def __init__(self, level1b, bname, param):
        self._name = str(bname)
        if self._name not in level1b.band_names():
            raise ValueError("invalid band name %s" % self._name)
        self._check_param(param)
        self._param = param

        vname, band_names = _BAND_VARNAMES[level1b.resolution(), self._name]
        self._index = None
        if band_names:
            self._index = band_names.index(self._name)
        self._super = super(Level1BBand, self)
        self._super.__init__(level1b, vname, self._index)

        if 'band_names' in self._attributes and ','.join(band_names) != self._attributes['band_names']:
            raise ValueError("Unexpected band order in image array")
        self._attributes = self._clean_attributes(self._attributes)

    @property
    def name(self):
        """The band name (str).
        """
        return self._name

    def _clean_attributes(self, attr):
        keys = [
            ("radiance_scales", "radiance_scale"),
            ("radiance_offsets", "radiance_offset"),
            ("reflectance_scales", "reflectance_scale"),
            ("reflectance_offsets", "reflectance_offset"),
            ("corrected_counts_scales", "corrected_counts_scale"),
            ("corrected_counts_offsets", "corrected_counts_offset"),
        ]
        for old, new in keys:
            if old in attr:
                if self._var.ndim == 2:
                    attr[new] = attr[old]
                else:
                    attr[new] = attr[old][self._index]
                del attr[old]
        return attr

    def _param_scale_offset(self, param):
        return self._attributes[param+"_scale"], self._attributes[param+"_offset"]

    def _check_param(self, param):
        if param not in _SUPPORTED_PARAMS:
            raise ValueError("invalid param %s" % param)
        if param == 'reflectance' and self._name in _1KM_EMISSIVE_BAND_NAMES:
            raise ValueError("Reflectance units is valid for bands 1-19, 26 only")
        if param == 'corrected_counts' and self._name in _1KM_EMISSIVE_BAND_NAMES:
            raise ValueError("Corrected counts is valid for bands 1-19, 26 only")
        if param == 'temperature' and self._name not in _1KM_EMISSIVE_BAND_NAMES:
            raise ValueError("Temperature units valid for bands 20-25, 27-36 only")

    def _convert(self, data, fromparam, toparam):
        self._check_param(fromparam)
        self._check_param(toparam)
        if fromparam == toparam:
            return data
        if fromparam == 'temperature':
            raise ValueError("conversion from temperature is not supported")

        if fromparam == 'raw':
            if toparam == 'temperature':
                scale, offset = self._param_scale_offset('radiance')
                data = scale * (data - float(offset))
                return _modis_bright(data, self._name, 1, self._level1b.is_terra())
            else:
                scale, offset = self._param_scale_offset(toparam)
                return scale * (data - float(offset))

        scale, offset = self._param_scale_offset(fromparam)
        return self._convert(data/float(scale)+offset, 'raw', toparam)

    def convert(self, data, fromparam, toparam):
        """Convert data from one param to another.

        Parameters
        ----------
        data : ndarray
            Data to be converted.
        fromparam : str
            Param of source data.
        toparam: str
            Param of the data to be returned.

        Returns
        -------
        new_data : ndarray
            Converted data.
        """
        fd = data
        if np.ma.isMaskedArray(data):
            mask = data.mask
            fill_value = self._convert(data.fill_value, fromparam, toparam)
            fd = data.filled()

        fd = self._convert(fd, fromparam, toparam)

        if np.ma.isMaskedArray(data):
            return np.ma.masked_array(fd, mask=mask, fill_value=fill_value)
        return fd

    def valid_range(self):
        """Returns the valid range of the data.

        Returns
        -------
        vr : ndarray of shape (2,)
            The valid range where the first element is the lower bound, and the
            second the upper bound.
        """
        vr = np.array(self._attributes['valid_range'])
        if self._param != 'raw':
            return self.convert(vr, 'raw', self._param)
        return vr

    def destripe(self, img, nbins=100, skipdet=None):
        """A wrapper around :func:`destripe`.

        The only difference is that resolution, and valid_range
        arguments is not needed.
        """
        return destripe(img, self._level1b.resolution(), self.valid_range(), nbins=nbins, skipdet=skipdet)

    def read(self, start=None, count=None, clean=False):
        """Read band data from file.

        Parameters
        ----------
        start : 2-tuple, optional
            Start of crop (top left corner).
        count : 2-tuple, optional
            Size of crop.
        clean : bool, optional
            Tries to fill values out of valid range in the image using
            :func:`Level1BBand.fill_invalid` and then destripes the image using
            :func:`Level1BBand.destripe`.

        Returns
        -------
        data : 2D masked ndarray
            Band data.
        """
        if clean and (start or count):
            raise ValueError("clean=True is not supported for a crop")

        img = self._super.read(start=start, count=count)
        img = self.convert(img, 'raw', self._param)
        if clean:
            return self.destripe(self.fill_invalid(img))
        return img

    def write(self, img, start=None):
        """Write band data to file.

        Parameters
        ----------
        img : 2D ndarray
            Data to be written.
        start : 2-tuple, optional
            Start of crop (top left corner).
        """
        if img.ndim != 2:
            raise ValueError('Image img has dimension %d' % img.ndim)
        img = self.convert(img, self._param, 'raw')
        return self._super.write(img, start=start)

def _get_detectors(bandnames, sd, name):
    # Band order: 1, 2, 3, ... 12, 13lo, 13hi, 14lo, 14hi, 15, ... , 36
    # 250m bands with 40 detectors each: 1,2
    # 500m bands with 20 detectors each: 3,4,5,6,7
    # The rest are 1km bands with 10 detectors each.
    ndet = np.array([40]*2 + [20]*5 + [10]*31)
    ind = np.concatenate(([0], np.cumsum(ndet)))

    bits = np.array(getattr(sd, name))
    dd = {}
    for i, name in enumerate(bandnames):
        dd[name], = np.where(bits[ind[i]:ind[i+1]] > 0)
    return dd

class Level1B(object):
    """Create an instance of Level 1B HDF product file.

    The MODIS Level 1B product is available in three different resolutions: 1
    kilometers, 500 meters, and 250 meters. The product filenames follow a
    naming convention. Consider the following filenames:

        1. MOD021KM.A2009057.1710.005.2009059004301.hdf
        2. MOD02HKM.A2009057.1710.005.2009059004301.hdf
        3. MOD02QKM.A2009057.1710.005.2009059004301.hdf
        4. MYD021KM.A2009057.1855.005.2009058162304.hdf
        5. MYD02HKM.A2009057.1855.005.2009058162304.hdf
        6. MYD02QKM.A2009057.1855.005.2009058162304.hdf

    1-3 are Terra granules because they begin with "MOD". 4-6 are Aqua granules
    because of the prefix "MYD". Also, the "1KM" (1,4), "HKM" (2,5), and "QKM"
    (3,6) in the filename indicates resolution 1 km, half km, and quarter km
    respectively. This class does not depends on filenames following this
    convention.

    Each file contains data for a number of bands. The QKM file contains bands
    1, and 2.  The HKM contains bands 1-7.  The 1KM contains the bands 1-12,
    13lo, 13hi, 14lo, 14hi, 15-36.

    Each band can be parameterized (param) with "raw", "radiance", "reflectance",
    "corrected_counts", or "temperature". The param "raw" refers to the
    unaltered version of the data as stored in an uint16 array in the HDF file.
    Some scale factors and offsets contained in the HDF metadata can be applied
    to the raw data to obtain radiance, reflectance, and corrected counts.
    The equation is ``scaled_data = scale * (raw_data - offset)``.
    Brightness temperature is obtained from radiances, and is only available for
    the emissive bands (bands 20-25, 27-36). Also, reflectance and corrected
    counts are only available for the non-emissive bands.

    Radiances have the units Watts/m^2/micrometer/steradian. Brightness
    Temperature is in Kelvin. Also note that reflectance are without solar
    zenith correction.

    A band is a 2D image. They typically have the shape (2030, 1354), (4060,
    2708), (8120, 5416) for 1KM, HKM, and QKM respectively.  They may contain
    missing pixels, which are masked using numpy masked array. They also have
    an associated valid range given in the HDF metadata, and values out of this
    range may appear in the image.

    In addition, each granule has latitude and longitude images with shape
    (406, 271), (2030, 1354), (2030, 1354) for 1KM, HKM, and QKM respectively.

    Examples
    --------
    In a typical interactive use, one-liner such as the following is used:

    >>> import modis
    >>> img = modis.Level1B("MYD021KM.A2011340.1045.005.2011357182859.hdf").radiance(6).read()
    >>> img.shape
    (2030, 1354)
    >>> img.min(), img.max()
    (0.22583057382144034, 178.29187760688365)

    Here is another example showing more detailed use:

    >>> import numpy as np
    >>> import modis
    >>> g = modis.Level1B("MYD021KM.A2011340.1045.005.2011357182859.hdf")
    >>> g.is_aqua()
    True
    >>> g.resolution()
    1000
    >>> g.band_names()
    ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13lo', '13hi', '14lo', '14hi', '15', '16', '17', '18', '19', '26', '20', '21', '22', '23', '24', '25', '27', '28', '29', '30', '31', '32', '33', '34', '35', '36']
    >>> b = g.radiance(6)
    >>> b.shape
    (2030, 1354)
    >>> b.name
    '6'
    >>> b.param
    'radiance'
    >>> b.valid_range()
    array([  0.        ,  89.15410135])
    >>> img = b.read()
    >>> np.sum(b.is_invalid(img))
    1099448
    >>> # fill invalid values and write it back into the HDF file
    >>> b.write(b.fill_invalid(img))
    >>> b.close()
    >>> g.close()


    Parameters
    ----------
    path : str
        Path name of MODIS Level 1B HDF file.
    mode : {'r', 'w'}, optional
        Open the file in read mode ('r') or read-write mode ('w').

    """
    def __init__(self, path, mode='r'):
        self._sd = ioutils.PyhdfWrapper(path, mode=mode)
        self._path = path

    def close(self):
        """Close the file.
        """
        return self._sd.close()

    @property
    def sd(self):
        """The underlying file object (pyhdf.SD.SD) used to access the file.
        """
        return self._sd.SD

    def is_terra(self):
        """Is this a Terra granule?

        Returns
        -------
        yes : bool
            True iff the file is from Terra.
        """
        return self._sd.isterra()

    def is_aqua(self):
        """Is this an Aqua granule?

        Returns
        -------
        yes : bool
            True iff the file is from Aqua.
        """
        return not self._sd.isterra()

    def resolution(self):
        """Returns the resolution of the granule.

        Returns
        -------
        res : {1000, 500, 250}
            The resolution in meters.
        """
        if self._sd.has_dataset('EV_1KM_Emissive'):
            return 1000
        elif self._sd.has_dataset('EV_500_RefSB'):
            return 500
        elif self._sd.has_dataset('EV_250_RefSB'):
            return 250
        else:
            raise ValueError("%s is not a Level 1B HDF file" % self._path)

    def band_names(self):
        """Returns a list of available band names.

        Returns
        -------
        names : list of str
            Band names.
        """
        return {
            1000 : _QKM_BAND_NAMES+_HKM_BAND_NAMES+_1KM_REF_BAND_NAMES+_1KM_EMISSIVE_BAND_NAMES,
            500  : _QKM_BAND_NAMES+_HKM_BAND_NAMES,
            250  : _QKM_BAND_NAMES,
        }[self.resolution()]

    def dead_detectors(self):
        return _get_detectors(self.band_names(), self._sd.SD, 'Dead Detector List')

    def noisy_detectors(self):
        return _get_detectors(self.band_names(), self._sd.SD, 'Noisy Detector List')

    def band(self, bname, param):
        """Create an instance of :class:`Level1BBand`.

        Parameters
        ----------
        bname : int or str
            Band name. If an int, attempt will be made to convert
            it to str by ``str(bname)``.
        param : str
            Param of the data.
        """
        return Level1BBand(self, bname, param)

    def raw(self, bname):
        """Returns an instance of :class:`Level1BBand`.

        Equivalent to calling :func:`Level1B.band` with param='raw'.

        Parameters
        ----------
        bname : int or str
            Band name. If an int, attempt will be made to convert
            it to str by ``str(bname)``.
        """
        return Level1BBand(self, bname, "raw")

    def radiance(self, bname):
        """Returns an instance of :class:`Level1BBand`.

        Equivalent to calling :func:`Level1B.band` with param='radiance'.

        Parameters
        ----------
        bname : int or str
            Band name. If an int, attempt will be made to convert
            it to str by ``str(bname)``.
        """
        return Level1BBand(self, bname, "radiance")

    def reflectance(self, bname):
        """Returns an instance of :class:`Level1BBand`.

        Equivalent to calling :func:`Level1B.band` with param='reflectance'.

        Parameters
        ----------
        bname : int or str
            Band name. If an int, attempt will be made to convert
            it to str by ``str(bname)``.
        """
        return Level1BBand(self, bname, "reflectance")

    def corrected_counts(self, bname):
        """Returns an instance of :class:`Level1BBand`.

        Equivalent to calling :func:`Level1B.band` with param='corrected_counts'.

        Parameters
        ----------
        bname : int or str
            Band name. If an int, attempt will be made to convert
            it to str by ``str(bname)``.
        """
        return Level1BBand(self, bname, "corrected_counts")

    def temperature(self, bname):
        """Returns an instance of :class:`Level1BBand`.

        Equivalent to calling :func:`Level1B.band` with param='temperature'.

        Parameters
        ----------
        bname : int or str
            Band name. If an int, attempt will be made to convert
            it to str by ``str(bname)``.
        """
        return Level1BBand(self, bname, "temperature")

    def latitude(self):
        """Returns latitude image for the granule, an instance of :class:`Level1BVariable`.

        Returns
        -------
        lat : 2d ndarray
            Latitude image.
        """
        return Level1BVariable(self, "Latitude", None)

    def longitude(self):
        """Returns longitude image for the granule, an instance of :class:`Level1BVariable`.

        Returns
        -------
        lon : 2d ndarray
            longitude image.
        """
        return Level1BVariable(self, "Longitude", None)
