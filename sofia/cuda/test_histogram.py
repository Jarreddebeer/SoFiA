import numpy as np
from numpy.testing import (assert_array_almost_equal)
from cffi import FFI
import math

ffi = FFI()

histogram_header = open('./histogram.h', 'r')
ffi.cdef(histogram_header.read())

C = ffi.dlopen('./histogram.so')

def C_histogram(ary, bins):
    histo = np.zeros(bins.shape, np.int32)
    ary = np.ravel(ary)
    ary_ptr = ffi.cast("float*", ary.ctypes.data)
    bins_ptr = ffi.cast("float*", bins.ctypes.data)
    histo_ptr = ffi.cast("int*", histo.ctypes.data)
    C.histogram(ary_ptr, bins_ptr, histo_ptr, len(ary), len(bins), len(histo))
    return histo[:-1]

class TestHistogram:

    def test_standard_bin(self):
        a = np.random.rand(500).astype(np.float32)
        bins = np.arange(0, 1, 0.05).astype(np.float32)

        histo = np.histogram(a, bins=bins)
        c_histo = C_histogram(a, bins)

        assert_array_almost_equal(histo[0], c_histo, decimal=3)


    def test_multi_array(self):
        a = np.random.rand(10, 30, 30).astype(np.float32)
        bins = np.arange(0, 1, 0.05).astype(np.float32)

        histo = np.histogram(a, bins=bins)
        c_histo = C_histogram(a, bins)

        assert_array_almost_equal(histo[0], c_histo, decimal=3)



    def test_negative_bin(self):
        mn = -0.0027436172
        # mx = 0.019781137
        a = np.random.rand(500).astype(np.float32)
        nrbins = max(100, int(math.ceil(float(np.array(a.shape).prod()) / 1e+5)))
        mx = abs(mn) / nrbins - 1e-12
        bins = np.arange(mn, mx, abs(mn) / nrbins).astype(np.float32)
        a = a * (mx - mn) - abs(mn)

        histo = np.histogram(a, bins=bins)
        c_histo = C_histogram(a, bins)

        assert_array_almost_equal(histo[0], c_histo, decimal=3)
