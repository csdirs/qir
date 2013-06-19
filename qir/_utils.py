
import numpy as np

def xslidingwin(im, wsize):
    """A generator that yields sliding windows of the image `im`.

    Parameters
    ----------
    im : 2-D or 3-D ndarray
        Image from which windows are obtained. If 3-D, the
        windows are computed with respect to the first two dimensions.
    wsize : 2-tuple
        Window size.

    """
    dy, dx = wsize
    nx = im.shape[1]-dx+1
    ny = im.shape[0]-dy+1

    for i in xrange(ny):
        for j in xrange(nx):
            yield im[i:i+dy, j:j+dx]

def fillinvalid(img, validrange=None, invalid=None, fillable=None, winsize=21, maxinvalid=0.5, pad=False):
    """Fill the invalid pixels in img indicated by either validrange
    or invalid. The filling is done by taking the average of the valid
    pixels in a 2D window to predict the center pixel.  The first
    window with at most maxinvalid fraction of invalid pixels out of
    sucessively large windows starting from shape (3,3) and ending
    with shape (winsize, winsize) is used. If no such window
    exists, the center pixel is not filled in.

    Parameters
    ----------
    img : 2d ndarray
        Image containing invalid pixels.
    validrange : 2-tuple, optional
        Values in the given range is considered valid.
    invalid : 2d ndarray, optional
        Boolean mask of the same shape as img indicating
        the location of invalid pixels.
    winsize : int, optional
        Maximum size of window. Must be odd.
    maxinvalid : float, optional
        If there are less than maxinvalid fraction of invalid
        pixels in the window, the window is used for filling.
    pad : bool, optional
        Pad the image so that winsize//2 pixels at the
        border are also filled.

    Returns
    -------
    new_img : 2d ndarray
        The filled image.
    new_invalid : 2d ndarray
        New invalid mask indicating which pixels were not filled.

    """
    if invalid is None:
        if len(validrange) != 2:
            raise ValueError("validrange has length %d" % (len(validrange),))
        invalid = (validrange[0] > img) | (img > validrange[1])
    if fillable is None:
        fillable = invalid.copy()
    if img.shape != invalid.shape:
        raise ValueError("img.shape %s != invalid.shape %s" % (img.shape, invalid.shape))
    if img.shape != fillable.shape:
        raise ValueError("img.shape %s != fillable.shape %s" % (img.shape, fillable.shape))
    if int(winsize)%2 != 1 or int(winsize) < 3:
        raise ValueError("winsize=%s must be an odd integer >= 3" % winsize)
    if 0 > maxinvalid or maxinvalid > 1:
        raise ValueError("maxinvalid=%s must be in the range [0,1]" % maxinvalid)

    invalid = np.array(invalid, dtype='bool')
    maxpad = min(int(winsize)//2, img.shape[0], img.shape[1])
    winsize = None  # winsize is wrong if img is smaller than winsize

    if pad:
        img = pad_image(img, width=maxpad)
        invalid = pad_image(invalid, width=maxpad)
        fillable = pad_image(fillable, width=maxpad)
    newinvalid = invalid.copy()
    newimg = img.copy()

    # Don't try to fill pixels near the border
    fillable = fillable.copy()
    fillable[:maxpad,:], fillable[-maxpad:,:] = False, False
    fillable[:,:maxpad], fillable[:,-maxpad:] = False, False

    for i, j in zip(*np.where(fillable)):
        for p in xrange(1, maxpad+1):
            ind = np.s_[i-p:i+p+1, j-p:j+p+1]
            win = img[ind]
            wininv = invalid[ind]
            if np.sum(wininv)/(2.0*p+1.0)**2 < maxinvalid:
                newimg[i,j] = win[~wininv].mean()
                newinvalid[i,j] = False
                break
    if pad:
        newimg = unpad_image(newimg, width=maxpad)
        newinvalid = unpad_image(newinvalid, width=maxpad)

    return newimg, newinvalid

def _get4widths(width=None, winsize=None):
    if width is None:
        if len(winsize) != 2:
            raise ValueError("winsize must be a 2-tuple")
        return (winsize[0]//2, winsize[0]//2, winsize[1]//2, winsize[1]//2)
    try:
        width = int(width)
        width = (width, width, width, width)
    except TypeError:
        width = tuple(width)
        if len(width) != 4:
            raise ValueError("width must be either an integer or a 4-tuple")
    if any([x < 0 for x in width]):
        raise ValueError("negative value in width=%s" % width)
    return width

# TODO: replace with numpy.pad (available in numpy >=1.7.0)
def pad_image(img, width=None, winsize=None):
    """Return the image resulting from padding width amount of pixels on
    each sides of the image img.  The padded values are mirror image with
    respect to the borders of img.

    Either width or winsize must be specified. Width can be an integer
    or a tuple (north, south, east, west). Winsize of (r, c) corresponds to
    a width of (r//2, r//2, c//2, c//2).

    """
    n, s, e, w = _get4widths(width=width, winsize=winsize)

    rows = []
    if n != 0:
        north = img[:n,:]
        rows.append(north[::-1,:])
    rows.append(img)
    if s != 0:
        south = img[-s:,:]
        rows.append(south[::-1,:])
    if len(rows) > 1:
        img = np.row_stack(rows)

    cols = []
    if w != 0:
        west = img[:,:w]
        cols.append(west[:,::-1])
    cols.append(img)
    if e != 0:
        east = img[:,-e:]
        cols.append(east[:,::-1])
    if len(cols) > 1:
        img = np.column_stack(cols)
    return img

def unpad_image(img, width=None, winsize=None):
    """Return unpadded image of img padded with :func:`pad_image`."""
    n, s, e, w = _get4widths(width=width, winsize=winsize)
    # index of -0 refers to the first element
    if s == 0:
        s = img.shape[0]
    else:
        s = -s
    if e == 0:
        e = img.shape[1]
    else:
        e = -e
    return img[n:s, w:e]
