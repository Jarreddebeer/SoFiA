from cffi import FFI
import numpy as np

ffi = FFI()

SCfinder_mem_header = open('./SCfinder_mem.h', 'r')
ffi.cdef(SCfinder_mem_header.read())

C = ffi.dlopen('./SCfinder_mem.so')

ary = np.array([
    [
        [1., 2., 3.],
        [4., 5., 6.],
        [7., 8., 9.]
    ],
    [
        [10., 20., 30.],
        [40., 50., 60.],
        [70., 80., 90.]
    ],
    [
        [100., 200., 300.],
        [400., 500., 600.],
        [700., 800., 900.]
    ],
], dtype = 'float32')
#

kernels_ary = [[ 0, 0, 0,'b'],[ 0, 0, 3,'b'],[ 0, 0, 7,'b'],[ 0, 0, 15,'b'],[ 3, 3, 0,'b'],[ 3, 3, 3,'b'],[ 3, 3, 7,'b'],[ 3, 3, 15,'b'],[ 6, 6, 0,'b'],[ 6, 6, 3,'b'],[ 6, 6, 7,'b'],[ 6, 6, 15,'b']]
for i in range(len(kernels_ary)):
    kern = kernels_ary[i]
    kern[3] = ord(kern[3])
kernels = np.array(kernels_ary, dtype = 'int32')


# for a in ary:

# ap_size = a.shape[0] * a.shape[1]
# print ap_size

ptr = ffi.cast("float*", ary.ctypes.data)
k_ptr = ffi.cast("int*", kernels.ctypes.data)

C.SCfinder_mem(ptr, ary.shape[0], ary.shape[1], ary.shape[2], k_ptr, kernels.shape[0])

#

print ary
