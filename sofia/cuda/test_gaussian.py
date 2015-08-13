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

# kernels = [[ 0, 0, 0,98],[ 0, 0, 3,98],[ 0, 0, 7,98],[ 0, 0, 15,98],[ 3, 3, 0,98],[ 3, 3, 3,98],[ 3, 3, 7,98],[ 3, 3, 15,98],[ 6, 6, 0,98],[ 6, 6, 3,98],[ 6, 6, 7,98],[ 6, 6, 15,98]]
kernels = [[ 4, 4, 0,98]]

def sumsq(a, b):
    return math.sqrt(((a - b)**2).sum())

def C_SCfinder_mem(input, kernel):
    kernel = np.array(kernel, dtype='int32')
    kernel_ptr = ffi.cast("int*", kernel.ctypes.data)
    input_ptr = ffi.cast("float*", input.ctypes.data)
    C.SCfinder_mem(input_ptr, input.shape[0], input.shape[1], input.shape[2], kernel_ptr, 1)

class TestGaussian:

    '''
    def test_gauss01(self):
        input = numpy.array([[1, 2, 3],
                             [2, 4, 6]], numpy.float32)
        output = ndimage.gaussian_filter(input, 0)
        assert_array_almost_equal(output, input)

    def test_array_3x1_a(self):
        input = numpy.array([
            [[1, 2, 3]]
        ], numpy.float32)

        output = ndimage.gaussian_filter(input, 3 / 2.355, truncate=1)
        C_SCfinder_mem(input, [3, 3, 3, 98])

        assert_array_almost_equal(output, input)

    '''
    def test_array_3x2_a(self):
        input = numpy.array([
            [[1, 2, 3, 4, 5]]

        ], numpy.float32)

        for kernel in kernels:
            input_copy = numpy.array(input)
            output = ndimage.gaussian_filter(input_copy, kernel[0] / 2.355, truncate=1)
            C_SCfinder_mem(input_copy, kernel)
            assert_array_almost_equal(output, input_copy)

    '''
    def test_array_2x3_b(self):
        input = numpy.array([
            [[1, 2, 3],
            [2, 4, 6]]
        ], numpy.float32)

        output = ndimage.gaussian_filter(input, 3 / 2.355, truncate=1)
        C_SCfinder_mem(input, [3, 3, 3, 98])

        print '------'
        print input
        print '------'
        print output
        print '------'

        assert_array_almost_equal(output, input)

    '''

    '''
    def test_gauss03(self):
        # single precision data"
        input = numpy.arange(100 * 100).astype(numpy.float32)
        input.shape = (100, 100)
        output = ndimage.gaussian_filter(input, [1.0, 1.0])

        assert_equal(input.dtype, output.dtype)
        assert_equal(input.shape, output.shape)

        # input.sum() is 49995000.0.  With single precision floats, we can't
        # expect more than 8 digits of accuracy, so use decimal=0 in this test.
        assert_almost_equal(output.sum(dtype='d'), input.sum(dtype='d'), decimal=0)
        assert_(sumsq(input, output) > 1.0)

    def test_gauss04(self):
        input = numpy.arange(100 * 100).astype(numpy.float32)
        input.shape = (100, 100)
        otype = numpy.float64
        output = ndimage.gaussian_filter(input, [1.0, 1.0],
                                                            output=otype)
        assert_equal(output.dtype.type, numpy.float64)
        assert_equal(input.shape, output.shape)
        assert_(sumsq(input, output) > 1.0)

    def test_gauss05(self):
        input = numpy.arange(100 * 100).astype(numpy.float32)
        input.shape = (100, 100)
        otype = numpy.float64
        output = ndimage.gaussian_filter(input, [1.0, 1.0],
                                                 order=1, output=otype)
        assert_equal(output.dtype.type, numpy.float64)
        assert_equal(input.shape, output.shape)
        assert_(sumsq(input, output) > 1.0)

    def test_gauss06(self):
        input = numpy.arange(100 * 100).astype(numpy.float32)
        input.shape = (100, 100)
        otype = numpy.float64
        output1 = ndimage.gaussian_filter(input, [1.0, 1.0],
                                                            output=otype)
        output2 = ndimage.gaussian_filter(input, 1.0,
                                                            output=otype)
        assert_array_almost_equal(output1, output2)

    '''
