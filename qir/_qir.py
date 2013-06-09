#!/usr/bin/env python

import numpy as np
from scipy.ndimage.filters import convolve

from . import _modis as modis
from ._utils import unpad_image, fillinvalid

Bands = np.array([6, 1, 3, 4, 5, 7], dtype='i') # Bands[0] will be restored
NDestripeBins = 100
WinSize = 5
PadSize = WinSize//2
FillWinSize = 21
NDetectors = 20

def verboseprint(*args):
    for s in args:
        print s,
    print

def noprint(*args):
    pass

VPrint = noprint

def check_globals():
    """Verify that global variable parameters are correct.
    """
    assert np.all((1 <= Bands) & (Bands <= 36))
    assert WinSize >= 3
    assert FillWinSize >= 1

def interp_nasa(img, baddets):
    """Fill in lines from bad detectors by linear interpolation
    of neighboring lines from good detectors.

    Parameters
    ----------
    img : 2d ndarray
        Band image. Modified in-place.
    baddets : 1d ndarray
        Bad detectors.

    Returns
    -------
    img : 2d ndarray
        Input img modified in-place.
    """
    gooddets = [d for d in xrange(NDetectors) if d not in baddets]

    left = -2*NDetectors + np.zeros(NDetectors)
    right = -2*NDetectors + np.zeros(NDetectors)
    for b in baddets:
        # Assume there is always a detector on the left in the current scan line.
        # There may not exist a detector on the right (last detector on 500m
        # resolution is bad).
        gr = [g for g in gooddets if g > b]
        if len(gr) != 0:
            right[b] = min(gr)
        left[b] = max([g for g in gooddets if g < b])

    for i in xrange(img.shape[0]//NDetectors):
        for b in baddets:
            k = i*NDetectors+b
            if right[b] < 0:
                img[k, :] = img[i*NDetectors+left[b], :]
            else:
                mu = (b-left[b])/float(right[b]-left[b])
                img[k, :] = (1-mu)*img[i*NDetectors+left[b], :] + mu*img[i*NDetectors+right[b], :]
    return img

def get_detector_mask(imgshape, dets):
    """Get mask indicating detector rows.

    Parameters
    ----------
    imgshape : 2-tuple
        Shape of the mask.
    dets : 1d ndarray
        List of detectors.

    Returns
    -------
    mask : 2d bool ndarray
        Mask.
    """
    mask = np.zeros(imgshape, dtype='bool')
    for d in dets:
        mask[d::NDetectors, :] = True
    return mask

def get_mask_wins(data, mask):
    """Get flattened windows data.

    Parameters
    ----------
    data : 3d ndarray
        All the bands stacked.
    mask : 2d bool ndarray
        Mask indicating missing values.

    Returns
    -------
    train : 2d ndarray
        Row i contains the neighboring pixels of target[i] from window i.
    target : 1d ndarray
        The targets for each window.
    """
    rows, cols = np.where(mask)
    target = data[rows, cols, 0].ravel()
    train = np.NaN + np.zeros((len(target), WinSize*WinSize*(len(Bands)-1)))

    k = 0
    for roff in xrange(-PadSize, PadSize+1):
        for coff in xrange(-PadSize, PadSize+1):
            train[:, k:k+(len(Bands)-1)] = data[rows+roff, cols+coff, 1:]
            k += len(Bands)-1
    return train, target

class LeastSquare(object):
    """Create an instance of least square predictor.

    Parameters
    ----------
    X : 2d ndarray
        Row i contains the neighboring pixels of Y[i] from window i.
    Y : 1d ndarray
        The targets for each window.
    """
    def __init__(self, X, Y):
        X = np.hstack((np.ones((X.shape[0],1)), X))
        self.alpha, _, _, _ = np.linalg.lstsq(X, Y)
        self.X = X
        self.Y = Y

    def getresidues(self):
        """Calculate least square residues.

        Returns
        -------
        residues : 1d ndarray
            Residues calculated from the training data.
        """
        return np.fabs(np.dot(self.X, self.alpha) - self.Y)

    def smallresidue(self):
        """Find if the least square residue is small.

        Returns
        -------
        small : 1d bool ndarray
            True iff the residue is small.
        """
        res = self.getresidues()
        return np.fabs(res - np.mean(res)) <= 2*np.std(res)

    def predict(self, X):
        """Predict values using learned parameters.

        Parameters
        ----------
        data : 3d ndarray
            Images from all bands.
        region : object
            Region instance.

        Returns
        -------
        img : 2d ndarray
            Predicted values in the region.
        """
        X = np.hstack((np.ones((X.shape[0],1)), X))
        return np.dot(X, self.alpha)

def alignhigh(x):
    return ((x+NDetectors-1)//NDetectors)*NDetectors

def alignlow(x):
    return (x//NDetectors)*NDetectors

def mask_bbox(mask):
    rind, cind = np.where(mask)
    return np.s_[
            alignhigh(np.min(rind)) : alignlow(np.max(rind)+1),
            np.min(cind) : np.max(cind)+1,
    ]

def read_mod02HKM(path, b6deaddets=None):
    """Read 500m resolution MODIS data. The image is destriped, and then
    values out of validrange are filled.

    Parameters
    ----------
    path : str
        Path to MODIS granule.
    b6deaddets : list
        List of band 6 dead detectors

    Returns
    -------
    data : 3d ndarray
        Data from all bands.
    validrange : 2d ndarray
        Nx2 array where N is the number of bands.
        Validrange[:,0] is the minimum valid value and
        validrange[:,1] is the maximum valid value.
    """
    hdf = modis.Level1B(path)
    imgshape = hdf.raw(6).shape
    data = np.zeros(imgshape + (len(Bands),))
    nightmask = np.zeros(imgshape, dtype='bool')
    validrange = np.zeros((len(Bands), 2))
    for i, band in enumerate(Bands):
        b = hdf.radiance(band)
        img = b.read()
        data[:,:,i] = img.data
        if np.any(img.mask):
            nightmask |= img.mask
        validrange[i,:] = b.valid_range()

    dayind = None
    if np.any(nightmask):
        dayind = mask_bbox(~nightmask)
        VPrint("Image shape:", imgshape)
        VPrint("Day slice:", dayind)
        if np.any(nightmask[dayind]):
            raise ValueError("non-contiguous day/night")
        data = data[dayind]

    deaddet = hdf.dead_detectors()
    if b6deaddets is not None:
        deaddet['6'] = b6deaddets
    dd = [deaddet[str(b)] for b in Bands]

    for i, band in enumerate(Bands):
        b = hdf.radiance(band)
        img = b.destripe(data[:,:,i], nbins=NDestripeBins, skipdet=dd[i])
        if len(dd[i]) == 0:
            fillmask = np.ones(img.shape, dtype='bool')
        else:
            VPrint("Band %s dead detectors: %s" % (band, dd[i]))
            fillmask = ~get_detector_mask(img.shape, dd[i])
        _img, _ = b.fill_invalid(
            img[fillmask].reshape((-1, img.shape[1])),
            winsize=FillWinSize,
            maxinvalid=0.5,
            pad=True,
        )
        img[fillmask] = _img.ravel()
        if band != 6 and len(dd[i]) > 0:
            img = interp_nasa(img, dd[i])
        data[:,:,i] = img

    return data, validrange, imgshape, dayind, dd

def argslidingwins(datashape, winsize, shiftsize):
    """Get location of sliding windows.

    Parameters
    ----------
    datashape : 2-tuple
        Shape of image.
    winsize : 2-tuple
        Shape of window.
    shiftsize : 2-tuple.
        Row and column shift sizes.

    Returns
    -------
    indices : list
        Indices to sliding windows.
    """
    lastrow, lastcol = datashape
    nrow, ncol = winsize

    args = []
    for rs in xrange(0, lastrow, shiftsize[0]):
        for cs in xrange(0, lastcol, shiftsize[1]):
            re, ce = rs+nrow, cs+ncol
            if re > lastrow:
                # first row must be detector aligned
                rs = (lastrow-nrow)//NDetectors*NDetectors
                re = lastrow
            if ce > lastcol:
                cs = lastcol-ncol
                ce = lastcol
            args.append((rs, cs, re, ce))
    return sorted(set(args))

def padded_crop(img, rect):
    """Crop from an image and then pad the cropped image using
    the original image if possible, otherwise using the reflection
    of crop along the edge.

    Parameters
    ----------
    img : 2d or 3d ndarray
        Image to crop from.
    rect : 4-tuple of int
        Defines the rectangle of the crop before padding:
        (start row, start column, end row, end column).

    Returns
    -------
    crop : 2d or 3d ndarray
        Cropped image.
    """
    rs, cs, re, ce = rect
    rs1, re1 = max(rs-PadSize, 0), min(re+PadSize, img.shape[0])
    cs1, ce1 = max(cs-PadSize, 0), min(ce+PadSize, img.shape[1])
    crop = img[rs1:re1, cs1:ce1]
    drs, dre = PadSize-(rs-rs1), PadSize-(re1-re)
    dcs, dce = PadSize-(cs-cs1), PadSize-(ce1-ce)

    v = []
    if drs > 0:
        v.append(crop[1:1+drs, :][::-1, :])
    v.append(crop)
    if dre > 0:
        v.append(crop[-dre-1:-1, :][::-1, :])
    if len(v) > 1:
        crop = np.row_stack(v)

    v = []
    if dcs > 0:
        v.append(crop[:, 1:1+dcs][:, ::-1])
    v.append(crop)
    if dce > 0:
        v.append(crop[:, -dce-1:-1][:, ::-1])
    if len(v) > 1:
        crop = np.column_stack(v)
    return crop

def set_pad(img, value):
    """Set image pad value.

    Parameters
    ----------
    img : 2d ndarray
        Padded image.
    value : float
        Value to be assigned to padded area.
    """
    img[:PadSize, :] = value
    img[-PadSize:, :] = value
    img[:, :PadSize] = value
    img[:, -PadSize:] = value

def modis_qir(datapath, b6deaddets=None, verbose=0):
    """Quatitative image restoration (QIR) of MODIS band 6.

    Parameters
    ----------
    datapath : str
        Path to a MODIS 500m resolution granule
    b6deaddets : list
        List of band 6 dead detectors

    Returns
    -------
    restored : 2d ndarray
        Image of QIR restored band 6 radiances.
    """
    global VPrint
    if verbose > 0:
        VPrint = verboseprint

    check_globals()
    data, validrange, imgshape, dayind, dd = \
            read_mod02HKM(datapath, b6deaddets=b6deaddets)
    VPrint("data shape:", data.shape, data.dtype)
    VPrint("WinSize =", WinSize)
    badmask = get_detector_mask(data[:,:,0].shape, dd[0])

    img = modis_qir_masked(data, validrange, badmask)
    if dayind is not None:
        fullimg = np.ma.zeros(imgshape)
        fullimg.mask = True
        fullimg[dayind] = img
        img = fullimg
    return img

def modis_qir_masked(data, validrange, badmask):
    """
    Parameters
    ----------
    data : 3d ndarray
        Data for all bands.
    validrange : 2d ndarray
        Nx2 array where N is the number of bands.
        Validrange[:,0] is the minimum valid value and
        validrange[:,1] is the maximum valid value.
    badmask : 2d bool ndarray
        Mask indicating missing values.

    Returns
    -------
    restored : 2d ndarray
        Image of QIR restored band 6 radiances.
    """
    origband = data[:,:,0].copy()
    goodmask = ~badmask

    # Only train (predict) on windows that are centered
    # on a good (dead) detector row,
    # and don't have a pixel with invalid value.
    validmask = (badmask | \
        ((validrange[0,0] <= data[:,:,0]) & (data[:,:,0] <= validrange[0,1]))) & \
        np.all((validrange[1:,0] <= data[:,:,1:]) & (data[:,:,1:] <= validrange[1:,1]), axis=2)
    vwinmask = (convolve(~validmask, np.ones((WinSize, WinSize)), mode='mirror') == 0)
    trainmask = ~badmask & vwinmask
    predictmask = badmask & vwinmask

    # Important: make sure bad detector pixels are not input to the algorithm
    data[badmask,0] = np.NaN

    restored = np.zeros_like(data[:,:,0])
    patchcount = np.zeros_like(data[:,:,0])

    patchsize = (min(10*NDetectors, data.shape[0]), min(10*NDetectors, data.shape[1]))
    patches = argslidingwins(data.shape[:2], patchsize, (patchsize[0]//2, patchsize[1]//2))
    i = 0
    for rect in patches:
        rs, cs, re, ce = rect
        VPrint("patch %4d/%d: (%3d, %3d) @ (%4d, %4d)" % (
                        i+1, len(patches), re-rs, ce-cs, rs, cs))
        crop = padded_crop(data, rect)
        tmask = np.copy(padded_crop(trainmask, rect))
        set_pad(tmask, False)
        pmask = np.copy(padded_crop(predictmask, rect))
        set_pad(pmask, False)

        if np.sum(tmask) == 0 or np.sum(pmask) == 0:
            continue
        # update tmask
        gX, gY = get_mask_wins(crop, tmask)
        ls = LeastSquare(gX, gY)
        tmask[tmask] = ls.smallresidue()
        if np.sum(tmask) == 0:
            continue

        gX, gY = get_mask_wins(crop, tmask)
        bX, _ = get_mask_wins(crop, pmask)
        ls = LeastSquare(gX, gY)
        crop[pmask, 0] = ls.predict(bX)

        crop = unpad_image(crop, width=PadSize)
        restored[rs:re, cs:ce] += crop[:,:,0]
        patchcount[rs:re, cs:ce] += np.ones_like(crop[:,:,0])
        i += 1

    restored /= patchcount.astype('f8')

    # Handle values out of valid range in restored image
    VPrint("Valid range before:", validrange[0,:])
    gooddata = origband[trainmask]
    std = np.std(gooddata)
    vrange = np.array([
        max(validrange[0,0], np.min(gooddata)-0.05*std),
        min(validrange[0,1], np.max(gooddata)+0.05*std),
    ])
    VPrint("Valid range after:", vrange)
    # We don't trust predicted values that are out of valid range,
    # so only fill pixels on good detector rows.
    _restored, _ = fillinvalid(restored[goodmask].reshape((-1, restored.shape[1])),
        validrange=vrange,
        winsize=FillWinSize, maxinvalid=0.5, pad=True)
    restored[goodmask] = _restored.ravel()

    # Not all invalid pixels may be filled, so return a masked array
    return np.ma.masked_where(
            np.isnan(restored) | (restored < vrange[0]) | (vrange[1] < restored),
            restored)
