from cffi import FFI
import numpy

ffi = FFI()

SCfinder_mem_header = open('./SCfinder_mem.h', 'r')
ffi.cdef(SCfinder_mem_header.read())

C = ffi.dlopen('./SCfinder_mem.so')

ary = numpy.array([
    [[1, 2, 3, 4, 5, 6, 7, 8],
     [1, 2, 3, 4, 5, 6, 7, 8],
     [1, 2, 3, 4, 5, 6, 7, 8],
     [1, 2, 3, 4, 5, 6, 7, 8],
     [1, 2, 3, 4, 5, 6, 7, 8],
     [1, 2, 3, 4, 5, 6, 7, 8],
     [1, 2, 3, 4, 5, 6, 7, 8],
     [1, 2, 3, 4, 5, 6, 7, 8] ]
])

kernel = numpy.array([3, 0, 0])

def C_gaussian_filter(ary, kernel):
    out = numpy.array(ary)
    ary_ptr = ffi.cast("double*", ary.ctypes.data)
    out_ptr = ffi.cast("double*", out.ctypes.data)
    C.test_cuda(ary_ptr, out_ptr, ary.shape[0], ary.shape[1], ary.shape[2], kernel[1], kernel[0])

C_gaussian_filter(ary, kernel)
