#include <stdlib.h>
#include <stdio.h>
#include <string.h>

extern "C" {
#include "gaussianSeparable_kernel.h"
}

#define BLOCKSIZE 32


#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}






__device__ double convolve_1d_gpu_kernel(double *subcube, double *weights_sm, int sy, int sx, int lw, int offset_multiplier, int min_clip, int max_clip) {

    // perform gaussian filter
    double sum = d_kernel[0] * subcube[ty][tx + kernel_r];
    for (int k = 1; k < 1 + kernel_r; k++) {
        sum += d_kernel[k] * (subcube[ty][tx + kernel_r - k] + subcube[ty][tx + kernel_r + k]);
    }

    double sum = weights[lw] * subcube[sy][sx];

    for (int i = 1; i < lw + 1; i++) {
        double weight = weights[lw + i];
        lo_val = subcube[sy][sx - i];
        hi_val = subcube[sy][sx + i];
        sum += weight * (lo_val + hi_val);
    }

    return sum;
}



__global__ void gaussian_filter_1d_gpu_kernel(double *d_in_cube, double *d_out_cube, double *d_weights, int lw, int stride) {

    int tx, ty, sx, sy, x, y, cube_x, cube_y, cube_idx;
    __shared__ double subcube[blockDim.y][blockDim.x + 2 * kernel_r];
    __shared__ double weights_sm[2 * lw + 1]

    tx = threadIdx.x;
    ty = threadIdx.y;
    tz = threadIdx.z;
    sx = lw + tx;
    sy = ty;
    x = blockDim.x * blockIdx.x + tx;
    y = blockDim.y * blockIdx.y + ty;
    z = blockDim.z * blockIdx.z + tz;
    cube_x = dimGrid.x * blockDim.x;
    cube_y = dimGrid.y * blockDim.y;
    cube_idx = (dimGrid.z * cube_x * cube_y) + (y * cube_x) + x;


    // copy weights into shared memory
    //////////////////////////////////

    if (tx < 2 * lw + 1) {
        weights_sm[tx] = d_weights[tx];
    }

    // copy pixel into subcube shared memory
    ////////////////////////////////////////

    // TODO: generalize filtering, hardcoding x filtering for now.

    subcube[sy][sx] = d_in_cube[cube_idx];
    // copy window into shared memory
    if (tx == 0) { // left boundary
        if (x == 0) { // cube hard boundary, pad with zero
            for (int i = 1; i <= lw; i++) {
                subcube[sy][sx - i] = 0; // TODO: pad the cube beforehand rather
            }
        } else { // soft boundary, pad from cube pixels
            for (int i = 1; i <= lw; i++) {
                subcube[sy][sx - i] = d_in_cube[cube_idx - i];
            }
        }

    } else if (tx == blockDim.x - 1) { // right boundary
        if (x == cube_x - 1) { // cube hard boundary, pad with zero
            subcube[sy][sx + i] = 0;
        } else { // soft boundary, pad from cube pixels
            subcube[sy][sx + i] = d_in_cube[cube_idx + i];
        }
    }

    __syncthreads();

    // perform convolution
    //////////////////////

    int max_clip = cube_idx - x + cube_x;
    int min_clip = max_clip - cube_x;

    d_out_cube[cube_idx] =
        convolve_1d_gpu_kernel(d_in_cube, weights_sm, sy, sx, lw, stride, min_clip, max_clip);

    __syncthreads();

}




void gaussian_filter_1d(double *in_cube, double *out_cube, size_t cube_z, size_t cube_y, size_t cube_x, size_t ks, int switch_xy) {

    int sd = ((double) kx) / 2.355;
    int lw = (int) (sd * 4 + 0.5);
    int weights_size = sizeof(double) * (2 * lw + 1);

    // generate the gaussian weights
    // __device__ __constant__ double d_kernel[kernel_w];
    double *h_weights    = (double *) malloc(weights_size);

    h_weights[lw] = 1.0;
    double sum = 1.0;
    // generate gaussian row weights
    for (int i = 1; i < lw + 1; i++) {
        double tmp = exp(-0.5 * ((double) i * i) / (sd * sd))
        h_weights[lw + i] = tmp;
        h_weights[lw - i] = tmp;
        sum += 2.0 * tmp;
    }
    for (int i = 0; i < 2 * lw + 1; i++) {
        h_weights[i] /= sum;
    }

    // set up the kernel and perform the convolution

    double *d_in_cube = NULL;
    double *d_out_cube = NULL;

    int cube_size = cube_z * cube_y * cube_x * sizeof(double);

    cudaMalloc(&d_in_cube, cube_size);
    cudaMalloc(&d_out_cube, cube_size);
    cudaMalloc(&d_weights, weights_size);

    cudaMemcpy(d_in_cube, h_in_cube, cube_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_out_cube, h_out_cube, cube_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_weights, h_weights, weights_size, cudaMemcpyHostToDevice);
    // cudaMemcpyToSymbol(d_kernel, h_row_kernel, row_kernel_size);

    dim3 dimBlock = dim3(BLOCKSIZE, BLOCKSIZE, 1);
    dim3 dimGrid = dim3(
        ceil( ((int)cube_x) / (float) dimBlock.x),
        ceil( ((int)cube_y) / (float) dimBlock.y),
        ceil( ((int)cube_z) / (float) dimBlock.z)
    );

    gaussian_filter_1d_gpu_kernel<<<dimGrid, dimBlock>>>(d_in_cube, d_out_cube, weights, lw, stride);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    cudaMemcpy(h_out_cube, d_out_cube, cube_size, cudaMemcpyDeviceToHost);

}



void gaussian_filter_GPU(double *in_cube, double *out_cube, int cube_z, int cube_y, int cube_x, int ky, int kx) {

    /*
    if (ky > 0) {
        gaussian_filter_1d(in_cube, out_cube, cube_z, cube_y, cube_x, ky, 1);
    }
    */
    if (kx > 0) {
        gaussian_filter_1d(in_cube, out_cube, cube_z, cube_y, cube_x, kx, 0);
    }


}

