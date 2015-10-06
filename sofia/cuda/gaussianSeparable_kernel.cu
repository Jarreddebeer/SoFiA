#include <stdlib.h>
#include <stdio.h>
#include <string.h>

extern "C" {
#include "gaussianSeparable_kernel.h"
}

#define BLOCKSIZE 8
#define MAX_LW 257 // CONSTRAINT: max size of lw (window) is 256
__device__ __constant__ double d_weights[MAX_LW];


#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
   if (code != cudaSuccess) {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}


void copy3d(double *to, double *from, size_t cube_z, size_t cube_y, size_t cube_x) {
    #pragma omp parallel for
    for (size_t z = 0; z < cube_z; z++) {
        for (size_t y = 0; y < cube_y; y++) {
            for (size_t x = 0; x < cube_x; x++) {
                size_t cube_idx = (z * cube_y * cube_x) + (y * cube_x) + x;
                to[cube_idx] = from[cube_idx];
            }
        }
    }
}




__device__ double convolve_1d_gpu_kernel(double *subcube, int sidx, int lw, int sstride) {

    double sum = d_weights[lw] * subcube[sidx];
    double lo_val, hi_val, weight;

    for (int i = 1; i <= lw; i++) {
        weight = d_weights[lw + i];
        lo_val = subcube[sidx - i * sstride];
        hi_val = subcube[sidx + i * sstride];
        sum += weight * (lo_val + hi_val);
    }

    return sum;
}



__global__ void gaussian_filter_1d_gpu_kernel(double *d_in_cube, double *d_out_cube, int lw, int stride, int cube_z, int cube_y, int cube_x) {

    int tx, ty, sx, sy, x, y, cube_idx, sidx, sstride;
    extern __shared__ double subcube[];

    tx = threadIdx.x;
    ty = threadIdx.y;
    sx = lw + tx;
    sy = lw + ty;
    sstride = blockDim.x + 2 * lw;
    sidx = sy * sstride + sx;
    x = blockDim.x * blockIdx.x + tx;
    y = blockDim.y * blockIdx.y + ty;
    cube_idx = (blockIdx.z * cube_x * cube_y) + (y * cube_x) + x;

    // copy pixel into subcube shared memory
    ////////////////////////////////////////

    if (x >= cube_x || y >= cube_y) {
        subcube[sidx] = 0;
    } else {
        subcube[sidx] = d_in_cube[cube_idx];
    }

    // copy window into shared memory
    ////////////////////////////////////

    // X padding

    if (tx == 0) { // left boundary
        if (x == 0) { // cube hard boundary, pad with zero
            for (int i = 1; i <= lw; i++) {
                subcube[sidx - i] = 0;
            }
        } else { // soft boundary, pad from cube pixels
            for (int i = 1; i <= lw; i++) {
                subcube[sidx - i] = d_in_cube[cube_idx - i];
            }
        }

    } else if (tx == blockDim.x - 1) { // right boundary
        if (x == cube_x - 1) { // cube hard boundary, pad with zero
            for (int i = 1; i <= lw; i++) {
                subcube[sidx + i] = 0;
            }
        } else { // soft boundary, pad from cube pixels
            for (int i = 1; i <= lw; i++) {
                if (x + i < cube_x) { // the window column might be thicker than the remaining columns in the cube
                    subcube[sidx + i] = d_in_cube[cube_idx + i];
                } else {
                    subcube[sidx + i] = 0;
                }
            }
        }
    }

    // Y padding

    if (ty == 0) { // top boundary
        if (y == 0) { // cube hard boundary, pad with zero
            for (int i = 1; i <= lw; i++) {
                subcube[sidx - i * sstride] = 0;
            }
        } else { // soft boundary, pad from cube pixels
            for (int i = 1; i <= lw; i++) {
                subcube[sidx - i * sstride] = d_in_cube[cube_idx - i * stride];
            }
        }

    } else if (ty == blockDim.y - 1) { // bottom boundary
        if (y == cube_y - 1) { // cube hard boundary, pad with zero
            for (int i = 1; i <= lw; i++) {
                subcube[sidx + i * sstride] = 0;
            }
        } else { // soft boundary, pad from cube pixels
            for (int i = 1; i <= lw; i++) {
                if (y + i < cube_y) { // the window row might be thicker than the remaining rows in the cube
                    subcube[sidx + i * sstride] = d_in_cube[cube_idx + i * stride];
                } else {
                    subcube[sidx + i * sstride] = 0;
                }
            }
        }
    }

    // corner cases

    if (tx == 0 && ty == 0) { // TL
        for (int i = 0; i < lw; i++) {
            for (int j = 0; j < lw; j++) {
                subcube[i * sstride + j] = 0;
            }
        }

    } else if (tx == blockDim.x - 1 && ty == blockDim.y - 1) { // BR
        for (int i = lw + blockDim.y; i < lw + blockDim.y + lw; i++) {
            for (int j = lw + blockDim.x; j < lw + blockDim.x + lw; j++) {
                subcube[i * sstride + j] = 0;
            }
        }
    } else if (tx == 0 && ty == blockDim.y - 1) { // BL
        for (int i = lw + blockDim.y; i < lw + blockDim.y + lw; i++) {
            for (int j = 0; j < lw; j++) {
                subcube[i * sstride + j] = 0;
            }
        }

    } else if (tx == blockDim.x - 1 && ty == 0) { // TR
        for (int i = 0; i < lw; i++) {
            for (int j = lw + blockDim.x; j < lw + blockDim.x + lw; j++) {
                subcube[i * sstride + j] = 0;
            }
        }
    }

    __syncthreads();

    /*
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        printf("PRINTING GRID:\n");
        printf("lw: %d sstride_y: %d, blockDim.x: %d\n", lw, sstride, blockDim.x);
        for (int i = 0; i < 8 + 2 * lw; i++) {
            for (int j = 0; j < 8 + 2 * lw; j++) {
                printf("%f ", subcube[i * (8 + 2 * lw) + j]);
            }
            printf("\n");
        }
        printf("\n");
    }

    __syncthreads();
    */

    // perform convolution
    //////////////////////

    if (x < cube_x && y < cube_y) {
        if (stride == 1) { // need to differentiate between x and y convolution
            d_out_cube[cube_idx] = convolve_1d_gpu_kernel(subcube, sidx, lw, 1);
        } else {
            d_out_cube[cube_idx] = convolve_1d_gpu_kernel(subcube, sidx, lw, sstride);
        }
    }

}




void gaussian_filter_1d(double *h_in_cube, double *h_out_cube, int cube_z, int cube_y, int cube_x, int ks, int stride) {

    double sd = ((double) ks) / 2.355;
    int lw = (int) (sd * 4 + 0.5);

    int weights_size = sizeof(double) * MAX_LW;
    int subcube_size = (BLOCKSIZE + 2 * lw) * (BLOCKSIZE + 2 * lw) * sizeof(double);

    // generate the gaussian weights
    double *h_weights    = (double *) malloc(weights_size);

    h_weights[lw] = 1.0;
    double sum = 1.0;

    // generate gaussian row weights
    for (int i = 1; i <= lw; i++) {
        double tmp = exp(-0.5 * ((double) i * i) / (sd * sd));
        h_weights[lw + i] = tmp;
        h_weights[lw - i] = tmp;
        sum += 2.0 * tmp;
    }
    for (int i = 0; i <= 2 * lw; i++) {
        h_weights[i] /= sum;
    }

    // set up the kernel and perform the convolution

    double *d_in_cube = NULL;
    double *d_out_cube = NULL;

    int cube_size = cube_z * cube_y * cube_x * sizeof(double);

    cudaMalloc(&d_in_cube, cube_size);
    cudaMalloc(&d_out_cube, cube_size);

    gpuErrchk( cudaMemcpyToSymbol(d_weights, h_weights, sizeof(double) * MAX_LW) );
    gpuErrchk( cudaMemcpy(d_in_cube, h_in_cube, cube_size, cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy(d_out_cube, h_out_cube, cube_size, cudaMemcpyHostToDevice) );

    dim3 dimBlock = dim3(BLOCKSIZE, BLOCKSIZE, 1);
    dim3 dimGrid = dim3(
        ceil( ((int)cube_x) / (float) dimBlock.x),
        ceil( ((int)cube_y) / (float) dimBlock.y),
        ceil( ((int)cube_z) / (float) dimBlock.z)
    );

    gaussian_filter_1d_gpu_kernel<<<dimGrid, dimBlock, subcube_size>>>(d_in_cube, d_out_cube, lw, stride, cube_z, cube_y, cube_x);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    cudaMemcpy(h_out_cube, d_out_cube, cube_size, cudaMemcpyDeviceToHost);

}



void gaussian_filter_GPU(double *in_cube, double *out_cube, int cube_z, int cube_y, int cube_x, int ky, int kx) {

    int stride;

    if (ky > 0) {
        stride = cube_x;
        gaussian_filter_1d(in_cube, out_cube, cube_z, cube_y, cube_x, ky, stride);
        copy3d(in_cube, out_cube, cube_z, cube_y, cube_x);
    }

    if (kx > 0) {
        stride = 1;
        gaussian_filter_1d(in_cube, out_cube, cube_z, cube_y, cube_x, kx, stride);
        copy3d(in_cube, out_cube, cube_z, cube_y, cube_x);
    }

}

