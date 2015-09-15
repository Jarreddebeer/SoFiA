import numpy
from numpy.testing import (assert_array_almost_equal)
import scipy.ndimage as ndimage
from cffi import FFI


ffi = FFI()

SCfinder_mem_header = open('./SCfinder_mem.h', 'r')
ffi.cdef(SCfinder_mem_header.read())

C = ffi.dlopen('./SCfinder_mem.so')

def C_uniform_filter_1D(ary, kz):
    out = numpy.array(ary)
    ary_ptr = ffi.cast("double*", ary.ctypes.data)
    out_ptr = ffi.cast("double*", out.ctypes.data)
    C.uniform_filter_1d(ary_ptr, out_ptr, ary.shape[0], ary.shape[1], ary.shape[2], kz)
    return out


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
        ], numpy.double)

        for kernel in kernels:
            input_copy = numpy.array(input)

            kz = kernel[2]
            output = ndimage.uniform_filter1d(input_copy, kz, axis=0, mode='constant')
            output_c = C_uniform_filter_1D(input_copy, kz)

            assert_array_almost_equal(output, output_c, decimal=3)

    def test_array_nxn(self):

        kernels = [[ 0, 0, 1,98],[ 0, 0, 3,98],[ 0, 0, 7,98],[ 0, 0, 15,98]]

        input = numpy.arange(15 * 10 * 10).astype(numpy.double)
        input.shape = (15, 10, 10)

        print(input)

        for kernel in kernels:
            input_copy = numpy.array(input)
            kz = kernel[2]

            output = ndimage.uniform_filter1d(input_copy, kz, axis=0, mode='constant')
            output_c = C_uniform_filter_1D(input, kz)

            assert_array_almost_equal(output, output_c, decimal=2)
