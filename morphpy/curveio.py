from svgpathtools import svg2paths
import numpy as np


def readsvgcurve(filename, T=100):
    paths, attributes = svg2paths(filename)
    s = np.linspace(0, 1, T, endpoint=True)
    coords = np.array([paths[0].point(ii) for ii in s], dtype=complex)
    X = np.stack((coords.real, coords.imag))
    return X


def readsvgcurvelist(filelist, T=100):
    with open(filelist) as f:
        filenames = f.readlines()
    filenames = [ii.strip() for ii in filenames]
    Xarray = np.array([readsvgcurve(fname, T) for fname in filenames], dtype=float)
    return Xarray






