#!/usr/bin/env python

import argparse
import modis
import numpy as np
import os.path
import shutil
import sys
from qir import modis_qir

def get_args():
    parser = argparse.ArgumentParser(description="""
        Quantitatively restore band 6 of a MODIS granule.
        The restored band 6 radiances will be saved as a MODIS Level 1B HDF file
        to the file named RESTORED.""")
    parser.add_argument('granule',
        help="""500m resolution MODIS granule filename.
        Must follow MODIS file naming convension.""")
    parser.add_argument('-d', dest='detectors',
        help="""Comma separated list of band 6 dead detectors
        that overrides the list in the HDF metadata.
        The detectors are between 0 and 19 inclusively.""")
    parser.add_argument('-o', dest='restored',
        help="""output filename of the restored band 6 image saved as MODIS Level 1B HDF file.
        The default is the granule name with the prefix "QIR." added.
        For example, running on granule
        MYD02HKM.A2010317.1935.005.2010318165656.hdf
        will save the restored image at
        QIR.MYD02HKM.A2010317.1935.005.2010318165656.hdf""")
    return parser.parse_args()

def main():
    args = get_args()

    if not args.restored:
        _, filename = os.path.split(args.granule)
        args.restored = "QIR."+filename
    if os.path.exists(args.restored):
        print >>sys.stderr, "%s already exists" % args.restored
        sys.exit(2)
    dets = None
    if args.detectors is not None:
        dets = np.array(sorted(set(int(d)
            for d in args.detectors.strip().split(','))), dtype='i')
        if np.any((dets < 0) | (19 < dets)):
            print >>sys.stderr, "invalid detector list:", args.detectors
            sys.exit(2)

    restored = modis_qir(args.granule, b6deaddets=dets).astype('f8')

    shutil.copyfile(args.granule, args.restored)
    g = modis.Level1B(args.restored, mode='w')
    try:
        b = g.radiance(6)
        b.write(restored)
        b.close()
    finally:
        g.close()

if __name__ == '__main__':
    main()
