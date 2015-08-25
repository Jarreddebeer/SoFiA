import numpy
from numpy.testing import (assert_array_almost_equal)
import scipy.ndimage as ndimage
import math
from cffi import FFI
import numpy as np


ffi = FFI()

SCfinder_mem_header = open('./SCfinder_mem.h', 'r')
ffi.cdef(SCfinder_mem_header.read())

C = ffi.dlopen('./SCfinder_mem.so')

class TestUniform:

    def test_array_3x1(self):

        kernels = [[0, 0, 3, 98]]

        input = numpy.array([
            [[1],
             [1]],
            [[1],
             [1]],
            [[1],
             [1]]
        ], numpy.float32)

        for kernel in kernels:
            input_copy = numpy.array(input)
            kz = kernel[2]
            output = ndimage.uniform_filter1d(input_copy, kz, axis=0, mode='constant')

            input_ptr = ffi.cast("float*", input_copy.ctypes.data)
            C.uniform_filter_1d(input_ptr, input.shape[0], input.shape[1], input.shape[2], kz)


            assert_array_almost_equal(output, input_copy, decimal=3)

    def test_array_nxn(self):

        kernels = [[ 0, 0, 1,98],[ 0, 0, 3,98],[ 0, 0, 7,98],[ 0, 0, 15,98]]

        input = numpy.arange(15 * 10 * 10).astype(numpy.float32)
        input.shape = (15, 10, 10)

        print(input)

        for kernel in kernels:
            input_copy = numpy.array(input)
            kz = kernel[2]
            output = ndimage.uniform_filter1d(input_copy, kz, axis=0, mode='constant')

            input_ptr = ffi.cast("float*", input_copy.ctypes.data)
            C.uniform_filter_1d(input_ptr, input.shape[0], input.shape[1], input.shape[2], kz)

            print('===========')
            print(output)
            print('-----------')
            print(input_copy)
            print('===========')

            assert_array_almost_equal(output, input_copy, decimal=2)

