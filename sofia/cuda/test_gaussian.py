import numpy
from numpy.testing import (assert_array_almost_equal)
import scipy.ndimage as ndimage
import math
from cffi import FFI


ffi = FFI()

SCfinder_mem_header = open('./SCfinder_mem.h', 'r')
ffi.cdef(SCfinder_mem_header.read())

C = ffi.dlopen('./SCfinder_mem.so')


def C_gaussian_filter(ary, kernel):
    # out = numpy.array(ary)
    out = numpy.array((1, 1, 1))
    ary_ptr = ffi.cast("double*", ary.ctypes.data)
    out_ptr = ffi.cast("double*", out.ctypes.data)
    C.test_cuda(ary_ptr, out_ptr, ary.shape[0], ary.shape[1], ary.shape[2], kernel[1], kernel[0])

'''
def C_SCfinder_mem(cube, kernels):
    kernels = numpy.array(kernels, dtype='int32')
    cube_ptr = ffi.cast("double*", cube.ctypes.data)
    kernel_ptr = ffi.cast("int*", kernels.ctypes.data)
    C.SCfinder_mem(cube_ptr, cube.shape[0], cube.shape[1], cube.shape[2], kernel_ptr, len(kernels))
'''

class TestGaussian:

    def test_array_3x1(self):

        # kernels = [[0, 0, 0, 98], [1, 0, 0, 98], [0, 1, 0, 98], [1, 1, 0, 98], [2, 1, 0, 98], [1, 2, 0, 98], [2, 2, 0, 98]]
        kernels = [[1, 0, 0, 98]]

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

        ], numpy.double)

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

        input = numpy.arange(10 * 50 * 50).astype(numpy.double)
        input.shape = (10, 50, 50)

        for kernel in kernels:
            input_copy = numpy.array(input)
            kx = kernel[0] / 2.355
            ky = kernel[1] / 2.355
            kz = 0
            output = ndimage.gaussian_filter(input_copy, [kz, ky, kx], mode='constant', truncate=4)
            C_gaussian_filter(input_copy, kernel)
            assert_array_almost_equal(output, input_copy, decimal=2)
