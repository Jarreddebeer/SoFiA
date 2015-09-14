#! /usr/bin/env python

import numpy as np
import math as mt
import scipy as sp
from scipy import optimize
from time import time
from cffi import FFI


ffi = FFI()

histogram_header = open('./sofia/cuda/histogram.h', 'r')
ffi.cdef(histogram_header.read())

C = ffi.dlopen('./sofia/cuda/histogram.so')


def C_histogram(ary, bins):
    histo = np.zeros(bins.shape, np.int32)
    ary = np.ravel(ary)
    ary_ptr = ffi.cast("float*", ary.ctypes.data)
    bins_ptr = ffi.cast("double*", bins.ctypes.data)
    histo_ptr = ffi.cast("int*", histo.ctypes.data)
    C.histogram(ary_ptr, bins_ptr, histo_ptr, len(ary), len(bins), len(histo))
    return histo[:-1]



def GaussianNoise(F, N0, s0):
    return N0 * np.exp( -F**2 / 2 / s0**2)

def GetRMS(cube, rmsMode='negative', zoomx=1, zoomy=1, zoomz=10000, nrbins=10000, verbose=0, min_hist_peak=0.05):

    sh=cube.shape

    if len(sh) == 2:
        # add an extra dimension to make it a 3d cube
        cube = np.array([cube])

    sh=cube.shape

    x0 = int( mt.ceil((1 - 1. / zoomx) * sh[2] / 2) )
    x1 = int( mt.floor((1 + 1. / zoomx) * sh[2] / 2) ) + 1
    y0 = int( mt.ceil((1 - 1. / zoomy) * sh[1] / 2) )
    y1 = int( mt.floor((1 + 1. / zoomy) * sh[1] / 2) ) + 1
    z0 = int( mt.ceil((1 - 1. / zoomz) * sh[0] / 2) )
    z1 = int( mt.floor((1 + 1. / zoomz) * sh[0] / 2) ) + 1

    if rmsMode == 'negative':

        binning_start = time()
        nrbins = max(
            100,
            int( mt.ceil( float(np.array(cube.shape).prod()) / 1e+5 ) )
        )
        cubemin = np.nanmin(cube)
        bins = np.arange(cubemin, abs(cubemin) / nrbins - 1e-12, abs(cubemin) / nrbins)
        fluxval = (bins[:-1] + bins[1:]) / 2
        histogram_1_start = time()
        # rmshisto = np.histogram(cube[z0:z1, y0:y1, x0:x1], bins=bins)[0]
        rmshisto = C_histogram(cube[z0:z1, y0:y1, x0:x1], bins)
        print('RMS - binning time - np.histogram (1) ', time() - histogram_1_start)

        nrsummedbins = 0

        while rmshisto[-nrsummedbins - 1:].sum() < min_hist_peak * rmshisto.sum():
            nrsummedbins += 1

        if nrsummedbins:
            nrbins /= (nrsummedbins + 1)
            bins = np.arange(cubemin, abs(cubemin) / nrbins - 1e-12, abs(cubemin) / nrbins)
            fluxval = (bins[:-1] + bins[1:]) / 2
            histogram_2_start = time()
            # rmshisto = np.histogram(cube[z0:z1, y0:y1, x0:x1], bins=bins)[0]
            rmshisto = C_histogram(cube[z0:z1, y0:y1, x0:x1], bins)
            print('RMS - binning time - np.histogram (2) ', time() - histogram_2_start)

        print('RMS - binning time: ', time() - binning_start)


        curve_fit_start = time()
        rms = abs(
            optimize.curve_fit(
                GaussianNoise,
                fluxval,
                rmshisto,
                p0 = [rmshisto.max(), -fluxval[rmshisto < rmshisto.max() / 2].max() * 2 / 2.355]
            )[0][1]
        )
        print('RMS - curve fitting time: ', time() - curve_fit_start)

    elif rmsMode == 'mad':
        rms = sp.stats.nanmedian(
            abs(
                cube[z0:z1, y0:y1, x0:x1] - sp.stats.nanmedian(cube[z0:z1, y0:y1, x0:x1], axis = None)
            ),
            axis = None
        ) / 0.6745


    elif rmsMode == 'std':
        rms = np.nanstd(cube[z0:z1, y0:y1, x0:x1], axis = None, dtype = np.float64)

    return rms
