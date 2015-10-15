from cffi import FFI
import numpy
from numpy.testing import assert_array_almost_equal
import scipy.ndimage as ndimage
from time import time

ffi = FFI()

SCfinder_mem_header = open('./SCfinder_mem.h', 'r')
ffi.cdef(SCfinder_mem_header.read())

C = ffi.dlopen('./SCfinder_mem.so')

def C_uniform_filter_cuda(ary, kernel):
    out = numpy.zeros((1, 1, 1)) # not used
    ary_ptr = ffi.cast("float*", ary.ctypes.data)
    out_ptr = ffi.cast("float*", out.ctypes.data)
    C.test_cuda_uniform(ary_ptr, out_ptr, ary.shape[0], ary.shape[1], ary.shape[2], kernel[2])


class TestCuda:

    def test_random_large(self):

        ary = numpy.random.random((660, 320, 320)).astype(numpy.float32)
        # ary = numpy.array([
        #     [[1]],
        #     [[2]],
        #     [[3]],
        #     [[4]],
        #     [[5]],
        #     [[6]],
        #     [[7]],
        #     [[8]],
        # ]).astype(numpy.float32)
        kernel = numpy.array([0, 0, 3]) # [x, y, z]
        ary_copy = numpy.array(ary).astype(numpy.float32)

        C_uniform_filter_cuda(ary, kernel)
        nmpy_out = ndimage.uniform_filter1d(ary_copy, kernel[2], axis=0, mode='constant')

        # print "CUDA:"
        # for i in range(5):
        #     start = time()
        #     C_uniform_filter_cuda(ary, kernel)
        #     print time() - start

        # print "NUMPY:"
        # for i in range(5):
        #     start = time()
        #     nmpy_out = ndimage.uniform_filter1d(ary, kernel[2], axis=0, mode='constant')
        #     print time() - start

        # print "CUDA"
        # print ary
        # print "NMPY"
        # print nmpy_out


        # for z in range(12):
        #     for y in range(315, 320):
        #         for x in range(315, 320):
        #             print abs(ary[z][y][x] - nmpy_out[z][y][x]),
        #         print
        #     print
        #     print

        # print
        # print

        # for z in range(12):
        #     print 'z level:', z, 'cuda: ', ary[z][319][319], 'nmpy: ', nmpy_out[z][319][319], 'original: ', ary_copy[z][319][319]

        print assert_array_almost_equal(ary, nmpy_out, decimal=2)
        # print assert_array_almost_equal([1], [2], decimal=2)

# tc = TestCuda()
# tc.test_random_large()
