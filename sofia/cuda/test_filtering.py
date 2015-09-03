import numpy
from numpy.testing import (assert_array_almost_equal)
import scipy.ndimage as nd
from cffi import FFI

ffi = FFI()

SCfinder_mem_header = open('./SCfinder_mem.h', 'r')
ffi.cdef(SCfinder_mem_header.read())

C = ffi.dlopen('./SCfinder_mem.so')


def C_SCfinder_mem(cube, kernels):
    kernels = numpy.array(kernels, dtype='int32')
    cube_ptr = ffi.cast("float*", cube.ctypes.data)
    kernel_ptr = ffi.cast("int*", kernels.ctypes.data)
    C.SCfinder_mem(cube_ptr, cube.shape[0], cube.shape[1], cube.shape[2], kernel_ptr, len(kernels))

class TestFinder:

    def test_array_nxn(self):

        kernels = [[ 0, 0, 0,98],[ 0, 0, 3,98],[ 0, 0, 7,98],[ 0, 0, 15,98],[ 3, 3, 0,98],[ 3, 3, 3,98],[ 3, 3, 7,98],[ 3, 3, 15,98],[ 6, 6, 0,98],[ 6, 6, 3,98],[ 6, 6, 7,98],[ 6, 6, 15,98]]

        input = numpy.arange(10 * 50 * 50).astype(numpy.float32)
        input.shape = (10, 50, 50)

        ##### SoFiA loop

        for kernel in kernels:
            input_copy_py = numpy.array(input)
            input_copy_c = numpy.array(input)
            kx = kernel[0] / 2.355
            ky = kernel[1] / 2.355
            kz = kernel[2]

            input_copy_py = nd.gaussian_filter(input_copy_py, [0, ky, kx], mode='constant', truncate=4)
            if kz:
                input_copy_py = nd.uniform_filter1d(input_copy_py, kz, axis=0, mode='constant')

            ###### my implementation

            C_SCfinder_mem(input_copy_c, [kernel])

            ###### assertions

            assert_array_almost_equal(input_copy_py, input_copy_c, decimal=2)

