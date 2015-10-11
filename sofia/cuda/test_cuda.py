from cffi import FFI
import numpy
from numpy.testing import assert_array_almost_equal
import scipy.ndimage as ndimage
from time import time

ffi = FFI()

SCfinder_mem_header = open('./SCfinder_mem.h', 'r')
ffi.cdef(SCfinder_mem_header.read())

C = ffi.dlopen('./SCfinder_mem.so')

def C_gaussian_filter_cuda(ary, kernel):
    out = numpy.zeros((1, 1, 1)) # not used
    ary_ptr = ffi.cast("double*", ary.ctypes.data)
    out_ptr = ffi.cast("double*", out.ctypes.data)
    C.test_cuda(ary_ptr, out_ptr, ary.shape[0], ary.shape[1], ary.shape[2], kernel[1], kernel[0])

def C_gaussian_filter(ary, kernel):
    out = numpy.array(ary)
    ary_ptr = ffi.cast("double*", ary.ctypes.data)
    out_ptr = ffi.cast("double*", out.ctypes.data)
    C.gaussian_filter(ary_ptr, out_ptr, ary.shape[0], ary.shape[1], ary.shape[2], kernel[1], kernel[0])
    return out

class TestCuda:

    def test_10x10(self):

        '''
        ary = numpy.array([
            [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] ]
        ]).astype(numpy.double)
        kernel = numpy.array([3, 3, 0]) # [x, y, z]

        ary_copy = numpy.array(ary)

        C_gaussian_filter_cuda(ary, kernel)
        nmpy_out = ndimage.gaussian_filter(ary_copy, [0, kernel[1] / 2.355, kernel[0] / 2.355], mode='constant', truncate=4)

        print("CUDA_OUT (ARY)")
        print(ary)

        print("NUMPY OUT")
        print(nmpy_out)

        assert_array_almost_equal(ary, nmpy_out, decimal=2)

    def test_10x10x3(self):

        ary = numpy.array([

           [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]],

           [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]],

           [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]

        ]).astype(numpy.double)

        kernel = numpy.array([3, 3, 1]) # [x, y, z]

        ary_copy = numpy.array(ary)

        C_gaussian_filter_cuda(ary, kernel)
        nmpy_out = ndimage.gaussian_filter(ary_copy, [0, kernel[1] / 2.355, kernel[0] / 2.355], mode='constant', truncate=4)

        assert_array_almost_equal(ary, nmpy_out, decimal=2)
    '''

    def test_random_large(self):

        ary = numpy.random.random((2460, 320, 320)).astype(numpy.double)
        kernel = numpy.array([10, 10, 0]) # [x, y, z]
        # ary_copy = numpy.array(ary)

        print "CUDA:"
        for i in range(5):
            start = time()
            C_gaussian_filter_cuda(ary, kernel)
            print time() - start

        print "NUMPY:"
        for i in range(5):
            start = time()
            nmpy_out = ndimage.gaussian_filter(ary, [0, kernel[1] / 2.355, kernel[0] / 2.355], mode='constant', truncate=4)
            print time() - start

        # assert_array_almost_equal(ary, nmpy_out, decimal=2)
        # assert_array_almost_equal([1], [2], decimal=2)

tc = TestCuda()
tc.test_random_large()
