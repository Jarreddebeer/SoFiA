from cffi import FFI

ffi = FFI()

SCfinder_mem_header = open('./SCfinder_mem.h', 'r')
ffi.cdef(SCfinder_mem_header.read())

C = ffi.dlopen('./SCfinder_mem.so')


def C_test_cuda():
    C.test_cuda()

C_test_cuda()

'''
class TestGaussian:

    def test_array_nxn(self):
        C_test_cuda()
'''
