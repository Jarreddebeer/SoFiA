#!/bin/python

import scipy.ndimage as ndimage
import numpy as np
from time import time
from tikz import *
from cffi import FFI

# testing
# -------
# - We test the gaussian and uniform CUDA kernels
#   with random data arrays. Each separable pass is timed
# -----------
# constraints
# -----------
# - kernels have equivalent filter sizes
# - cubes are rectangular z = 2*x, and x = y
# - data is in floats
# -----------

# ===================
# CFFI initialization
# ===================

ffi = FFI()
SCfinder_mem_header = open('./SCfinder_mem.h', 'r')
ffi.cdef(SCfinder_mem_header.read())

C = ffi.dlopen('./SCfinder_mem.so')

def C_uniform_filter_cuda(ary, kernel):
    out = np.zeros((1, 1, 1)) # not used
    ary_ptr = ffi.cast("float*", ary.ctypes.data)
    out_ptr = ffi.cast("float*", out.ctypes.data)
    C.test_cuda_uniform(ary_ptr, out_ptr, ary.shape[0], ary.shape[1], ary.shape[2], kernel[2])

def C_uniform_filter_omp(ary, kernel):
    out = np.array(ary)
    ary_ptr = ffi.cast("float*", ary.ctypes.data)
    out_ptr = ffi.cast("float*", out.ctypes.data)
    C.uniform_filter_1d(ary_ptr, out_ptr, ary.shape[0], ary.shape[1], ary.shape[2], kernel[2])

def C_gaussian_filter_cuda(ary, kernel):
    out = np.zeros((1, 1, 1))
    ary_ptr = ffi.cast("float*", ary.ctypes.data)
    out_ptr = ffi.cast("float*", out.ctypes.data)
    C.test_cuda(ary_ptr, out_ptr, ary.shape[0], ary.shape[1], ary.shape[2], kernel[1], kernel[0])

def C_gaussian_filter_omp(ary, kernel):
    out = np.array(ary)
    ary_ptr = ffi.cast("float*", ary.ctypes.data)
    out_ptr = ffi.cast("float*", out.ctypes.data)
    C.gaussian_filter(ary_ptr, out_ptr, ary.shape[0], ary.shape[1], ary.shape[2], kernel[1], kernel[0])


# ====================
# Program logic (main)
# ====================
y_label = 'Time (s)'
x_label = 'Data width (px)'

gaussian_pic_orig = TikzPicture('gaussian_original', 'Original gaussian runtime', x_label, y_label)
gaussian_pic_omp  = TikzPicture('gaussian_omp',      'OMP Gaussian runtime', x_label, y_label)
gaussian_pic_cuda = TikzPicture('gaussian_cuda',     'CUDA Gaussian runtime', x_label, y_label)

uniform_pic_orig  = TikzPicture('uniform_original', 'Original Uniform runtime' , x_label, y_label)
uniform_pic_omp   = TikzPicture('uniform_omp',      'OMP Uniform runtime' , x_label, y_label)
uniform_pic_cuda  = TikzPicture('uniform_cuda',     'CUDA Uniform runtime' , x_label, y_label)

# kernel sizes
kernel_sizes  = [1, 3, 6, 15]
kernel_colors = ['blue', 'green', 'orange', 'red']
marks        = ['x', '*', 'triangle*', 'square*']

for t in range(len(kernel_sizes)):

    ks = kernel_sizes[t]
    kernel = [ks, ks, ks]
    col = kernel_colors[t]
    mark = marks[t]

    print 'running kernel ', kernel, '...'

    g_plot_orig = TikzPlot(ks, col, mark)
    g_plot_omp  = TikzPlot(ks, col, mark)
    g_plot_cuda = TikzPlot(ks, col, mark)

    u_plot_orig = TikzPlot(ks, col, mark)
    u_plot_omp  = TikzPlot(ks, col, mark)
    u_plot_cuda = TikzPlot(ks, col, mark)

    # run 5 iterations of each
    for r in range(5):

        # data cube sizes are multiples of 32
        for w in range(0, 321, 32):

            h = w * 2
            print 'generating data of size (', h, w, w, ')'
            data = np.random.random((h, w, w)).astype(np.float32)

            # -------------
            # time gaussian
            # -------------
            print 'timing gaussian...'

            # original
            print 'original...'
            gs = time()
            ndimage.gaussian_filter(data, [ks/2.355, ks/2.355, 0], mode='constant', truncate=4)
            g_plot_orig.add_point(w, time() - gs)
            print 'original ran in', time() - gs

            # omp
            print 'omp...'
            gs = time()
            C_gaussian_filter_omp(data, kernel)
            g_plot_omp.add_point(w, time() - gs)
            print 'omp ran in', time() - gs

            # cuda
            print 'cuda...'
            gs = time()
            C_gaussian_filter_cuda(data, kernel)
            g_plot_cuda.add_point(w, time() - gs)
            print 'cuda ran in', time() - gs

            # ------------
            # time uniform
            # ------------
            print 'timing uniform...'

            # original
            print 'original...'
            us = time()
            ndimage.uniform_filter1d(data, ks, axis=0, mode='constant')
            u_plot_orig.add_point(w, time() - us)
            print 'original ran in', time() - us

            # omp
            print 'omp...'
            us = time()
            C_uniform_filter_omp(data, kernel)
            u_plot_omp.add_point(w, time() - us)
            print 'omp ran in', time() - us

            # cuda
            print 'cuda...'
            us = time()
            C_uniform_filter_cuda(data, kernel)
            u_plot_cuda.add_point(w, time() - us)
            print 'cuda ran in', time() - us


    gaussian_pic_orig.add_plot(g_plot_orig)
    gaussian_pic_omp.add_plot(g_plot_omp)
    gaussian_pic_cuda.add_plot(g_plot_cuda)

    uniform_pic_orig.add_plot(u_plot_orig)
    uniform_pic_omp.add_plot(u_plot_omp)
    uniform_pic_cuda.add_plot(u_plot_cuda)

print 'finished timings.'

print 'generating output...'
gaussian_pic_orig.generate()
gaussian_pic_omp.generate()
gaussian_pic_cuda.generate()

uniform_pic_orig.generate()
uniform_pic_omp.generate()
uniform_pic_cuda.generate()
print 'done.'
