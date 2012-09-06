
from pyhdf import SD

class PyhdfDataset(object):
    def __init__(self, dataset):
        self._ds = dataset

    def getattr(self):
        return self._ds.attributes()

    def getdtype(self):
        return self._ds[0,0].dtype

    @property
    def ndim(self):
        return len(self._ds.dimensions())

    @property
    def shape(self):
        return tuple(self._ds.dim(i).length() for i in xrange(self.ndim))

    @property
    def SDS(self):
        return self._ds

    def readarray(self, chan, start=None, count=None):
        ndim = len(self._ds.dimensions())
        if ndim == 2:
            return self._ds.get(start=start, count=count)

        if chan < 0 or chan >= self._ds.dim(0).length():
            raise ValueError("channel index %d out of range" % chan)
        if ndim == 3:
            if start is None or count is None:
                return self._ds[int(chan),:,:]
            else:
                start = (chan,)+start
                count = (1,)+count
                mimage = self._ds.get(start=start, count=count)
                return mimage[0,:,:]
        raise Exception("unexpected dimension of data %d" % ndim)

    def writearray(self, chan, mimage, start=None):
        if start is not None:
            count = mimage.shape
        ndim = len(self._ds.dimensions())
        if ndim == 2:
            self._ds.set(mimage, start=start, count=count)
            return

        if chan < 0 or chan >= self._ds.dim(0).length():
            raise ValueError("channel index %d out of range" % chan)
        if ndim == 3:
            if start is None or count is None:
                self._ds[int(chan),:,:] = mimage
            else:
                start = (chan,)+start
                count = (1,)+count
                self._ds.set(mimage, start=start, count=count)
            return
        raise Exception("unexpected dimension of data %d" % ndim)

    def close(self):
        self._ds.endaccess()

class PyhdfWrapper(object):
    def __init__(self, path, mode='r'):
        if mode == 'r':
            self._hdf = SD.SD(path, SD.SDC.READ)
        elif mode == 'w':
            self._hdf = SD.SD(path, SD.SDC.WRITE)
        else:
            raise ValueError("Bad mode=%s" % mode)
        self._datasets = self._hdf.datasets().keys()

    @property
    def SD(self):
        return self._hdf

    def isterra(self):
        import re

        # CoreMetadata.0 has lots of info encoded in a complex format.
        # Punt parsing it, and just search for what we're looking for.
        coremeta = self._hdf.attributes()["CoreMetadata.0"]
        terra = re.search("MOD02[1HQ]KM", coremeta)
        aqua = re.search("MYD02[1HQ]KM", coremeta)

        if terra and not aqua:
            return True
        if aqua and not terra:
            return False
        raise Exception("Bad CoreMetadata.0 attribute in HDF file")

    def has_dataset(self, name):
        return name in self._datasets

    def get_dataset(self, name):
        if name not in self._datasets:
            raise ValueError("dataset %s not found" % name)
        return PyhdfDataset(self._hdf.select(name))

    def close(self):
        self._hdf.end()
