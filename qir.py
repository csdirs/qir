#!/usr/bin/env python

import numpy as np
import os
import os.path

import modis
from utils import pad_image, unpad_image, fillinvalid

Bands = np.array([6, 1, 3, 4, 5, 7], dtype='i') # Bands[0] will be restored
NDestripeBins = 100
Band5Fix = 'interp'
WinSize = 5
FillWinSize = 21
NDetectors = 20
BadDetectors = np.array([1, 4, 5, 9, 11, 12, 13, 14, 15, 17, 18, 19])
GoodDetectors = np.array([_b for _b in range(NDetectors) if not _b in BadDetectors])

def check_globals():
    """Verify that global variable parameters are correct.
    """
    assert np.all((1 <= Bands) & (Bands <= 36))
    assert WinSize >= 3
    assert FillWinSize >= 1
    assert np.all((0 <= BadDetectors) & (BadDetectors < NDetectors))
    assert np.all((0 <= GoodDetectors) & (GoodDetectors < NDetectors))

def modtypeof(filename):
    """Returns the type of modis granule based on the filename.

    Parameters
    ----------
    filename : str
        MODIS granule filename.

    Returns
    -------
    modtype : str
        Either "Aqua" or "Terra".
    """
    s = os.path.basename(filename)[:5]
    if s == "MOD02":
        return "Terra"
    elif s == "MYD02":
        return "Aqua"
    raise ValueError("bad granule name %s" % filename)

def interp_nasa(img, gooddets, baddets):
    """Fill in lines from bad detectors by linear interpolation
    of neighboring lines from good detectors.

    Parameters
    ----------
    img : 2d ndarray
        Band image. Modified in-place.
    goodets : 1d ndarray
        Good detectors.
    baddets : 1d ndarray
        Bad detectors.

    Returns
    -------
    img : 2d ndarray
        Input img modified in-place.
    """
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
    for roff in xrange(-WinSize//2+1, WinSize//2+1):
        for coff in xrange(-WinSize//2+1, WinSize//2+1):
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

def read_mod02HKM(path):
    """Read 500m resolution MODIS data. The image is destriped, and then
    values out of validrange are filled.

    Parameters
    ----------
    path : str
        Path to MODIS granule.

    Returns
    -------
    data : 3d ndarray
        Data from all bands.
    validrange : 2d ndarray
        Nx2 array where N is the number of bands.
        Validrange[:,0] is the minimum valid value and
        validrange[:,1] is the maximum valid value.
    """
    data = []
    validrange = np.zeros((len(Bands), 2))
    hdf = modis.Level1B(path)
    for i, band in enumerate(Bands):
        b = hdf.radiance(band)
        img = b.read()
        if np.any(img.mask):
            raise Exception("flags exist in band %d of granule %s" % (band, os.path.basename(path)))
        img = b.fill_invalid(
            b.destripe(img.data.astype('f8'), nbins=NDestripeBins),
            winsize=FillWinSize,
            maxinvalid=0.5,
            pad=True,
        )
        n = np.sum(b.is_invalid(img))
        if n > 0:
            raise Exception("%d values out of valid range in band %d" % (n, band))
        validrange[i,:] = b.valid_range()
        data.append(img)

    return np.dstack(data), validrange

def fix_band5(data):
    """Fix band 5 of in data--intended for Terra granules.

    Parameters
    ----------
    data : 3d ndarray
        Data from all bands. It is modified in-place.

    Returns
    -------
    data : 3d ndarray
        Data with band 5 fixed if present.
    """
    if not np.any(Bands == 5):
        return data

    if Band5Fix == 'remove':
        ind = np.where(Bands != 5)[0]
        data = data[:,:,ind]
    elif Band5Fix == 'interp':
        baddets = np.array([3], dtype='i')
        gooddets = np.array(range(0, 3)+range(4, 20), dtype='i')
        ind = np.where(Bands == 5)[0][0]
        data[:,:,ind] = interp_nasa(data[:,:,ind], gooddets, baddets)
    else:
        raise ValueError("Band5Fix == %s unknown" % Band5Fix)
    return data

def argslidingwins(datashape, winsize, shiftsize):
    """Get location of sliding windows.

    Parameters
    ----------
    datashape : 2-tuple
        Shape of image.
    winsize : 2-tuple
        Shape of window.
    shiftsize : 2-typle.
        Row and column shift sizes.

    Returns
    -------
    indices : list
        Indices to sliding windows.
    """
    lastrow, lastcol = datashape
    nrow, ncol = winsize

    args = []
    for row in xrange(0, lastrow, shiftsize[0]):
        for col in xrange(0, lastcol, shiftsize[1]):
            r0, c0 = row, col
            rl, cl = r0+nrow, c0+ncol
            if rl > lastrow:
                # first row must be detector aligned
                r0 = (lastrow-nrow)//NDetectors*NDetectors
                rl = lastrow
            if cl > lastcol:
                c0 = lastcol-ncol
                cl = lastcol
            args.append((r0, c0, rl, cl))
    return args

def set_pad(img, value):
    """Set image pad value.

    Parameters
    ----------
    img : 2d ndarray
        Padded image.
    value : float
        Value to be assigned to padded area.
    """
    img[:WinSize//2, :] = value
    img[-WinSize//2+1:, :] = value
    img[:, :WinSize//2] = value
    img[:, -WinSize//2+1:] = value

def modis_qir(datapath):
    """Quatitative image restoration (QIR) of MODIS band 6.

    Parameters
    ----------
    datapath : str
        Path to a MODIS 500m resolution granule

    Returns
    -------
    restored : 2d ndarray
        Image of QIR restored band 6 radiances.
    """
    check_globals()
    data, validrange = read_mod02HKM(datapath)
    if modtypeof(datapath) == 'Terra':
        data = fix_band5(data)
    print "data shape:", data.shape, data.dtype
    print "WinSize =", WinSize
    badmask = get_detector_mask(data[:,:,0].shape, BadDetectors)
    return modis_qir_masked(data, validrange, badmask)

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
    data = pad_image(data, width=WinSize//2)
    print "After pad, data shape:", data.shape

    badmask = pad_image(badmask, width=WinSize//2)
    goodmask = ~badmask
    validmask = np.all((validrange[:,0] <= data) & (data <= validrange[:,1]), axis=2)

    # Important: make sure bad detector pixels are not input to the algorithm
    data[badmask,0] = np.NaN

    restored = np.zeros_like(data[:,:,0])
    patchcount = np.zeros_like(data[:,:,0])

    patches = argslidingwins(data.shape[:2], (10*NDetectors, 200), (5*NDetectors, 100))
    hw = WinSize//2
    i = 0
    for r0, c0, r, c in patches:
        print "patch %4d/%d: (%3d, %3d) @ (%4d, %4d)" % (
                        i+1, len(patches), r-r0, c-c0, r0, c0)
        crop = data[r0:r, c0:c, :]
        gmask = np.copy(goodmask[r0:r, c0:c])
        set_pad(gmask, False)
        bmask = np.copy(badmask[r0:r, c0:c])
        set_pad(bmask, False)
        vmask = validmask[r0:r, c0:c]

        # update gmask
        gX, gY = get_mask_wins(crop, gmask)
        ls = LeastSquare(gX, gY)
        gmask[gmask] = vmask[gmask] & ls.smallresidue()

        gX, gY = get_mask_wins(crop, gmask)
        bX, _ = get_mask_wins(crop, bmask)
        ls = LeastSquare(gX, gY)
        crop[bmask, 0] = ls.predict(bX)

        crop = unpad_image(crop, width=WinSize//2)
        restored[r0+hw:r-hw, c0+hw:c-hw] += crop[:,:,0]
        patchcount[r0+hw:r-hw, c0+hw:c-hw] += np.ones_like(crop[:,:,0])
        i += 1

    data = unpad_image(data, width=WinSize//2)
    restored = unpad_image(restored, width=WinSize//2)
    patchcount = unpad_image(patchcount, width=WinSize//2)
    restored /= patchcount.astype('f8')

    # Handle values out of valid range in restored image
    print "Valid range before:", validrange[0,:]
    goodmask = unpad_image(goodmask, width=WinSize//2)
    gooddata = origband[goodmask]
    std = np.std(gooddata)
    vrange = np.array([
        max(validrange[0,0], np.min(gooddata)-0.05*std),
        min(validrange[0,1], np.max(gooddata)+0.05*std),
    ])
    print "Valid range after:", vrange
    restored, _ = fillinvalid(restored, validrange=vrange,
        winsize=FillWinSize, maxinvalid=0.5, pad=True)

    # Not all invalid pixels may be filled, so return a masked array
    return np.ma.masked_outside(restored, vrange[0], vrange[1])
