import numpy
from numpy.testing import (assert_equal, assert_array_almost_equal)
import scipy.ndimage as ndimage
import math
from cffi import FFI
import numpy as np


ffi = FFI()

SCfinder_mem_header = open('./SCfinder_mem.h', 'r')
ffi.cdef(SCfinder_mem_header.read())

C = ffi.dlopen('./SCfinder_mem.so')


def sumsq(a, b):
    return math.sqrt(((a - b)**2).sum())

def C_gaussian_filter(input, kernel):
    input_ptr = ffi.cast("float*", input.ctypes.data)
    C.gaussian_filter(input_ptr, input.shape[0], input.shape[1], input.shape[2], kernel[2], kernel[1], kernel[0])

class TestGaussian:

    def test_array_3x1(self):

        kernels = [[0, 0, 0, 98], [1, 0, 0, 98], [0, 1, 0, 98], [1, 1, 0, 98], [2, 1, 0, 98], [1, 2, 0, 98], [2, 2, 0, 98]]

        input = numpy.array([
            [[1, 2, 3, 4],
             [4, 5, 6, 5],
             [7, 8, 9, 6],
             [2, 3, 4, 5]],

            [[1, 2, 3, 4],
             [4, 5, 6, 5],
             [7, 8, 9, 6],
             [2, 3, 4, 5]],

            [[1, 2, 3, 4],
             [4, 5, 6, 5],
             [7, 8, 9, 6],
             [2, 3, 4, 5]],

            [[1, 2, 3, 4],
             [4, 5, 6, 5],
             [7, 8, 9, 6],
             [2, 3, 4, 5]]

        ], numpy.float32)

        for kernel in kernels:
            input_copy = numpy.array(input)
            kx = kernel[0] / 2.355
            ky = kernel[1] / 2.355
            kz = 0
            output = ndimage.gaussian_filter(input_copy, [kz, ky, kx], mode='constant', truncate=4)
            C_gaussian_filter(input_copy, kernel)
            assert_array_almost_equal(output, input_copy, decimal=3)

    def test_array_nxn(self):

        kernels = [[ 0, 0, 0,98],[ 0, 0, 3,98],[ 0, 0, 7,98],[ 0, 0, 15,98],[ 3, 3, 0,98],[ 3, 3, 3,98],[ 3, 3, 7,98],[ 3, 3, 15,98],[ 6, 6, 0,98],[ 6, 6, 3,98],[ 6, 6, 7,98],[ 6, 6, 15,98]]

        input = numpy.arange(10 * 50 * 50).astype(numpy.float32)
        input.shape = (10, 50, 50)

        for kernel in kernels:
            input_copy = numpy.array(input)
            kx = kernel[0] / 2.355
            ky = kernel[1] / 2.355
            kz = 0
            output = ndimage.gaussian_filter(input_copy, [kz, ky, kx], mode='constant', truncate=4)
            C_gaussian_filter(input_copy, kernel)
            assert_array_almost_equal(output, input_copy, decimal=2)
