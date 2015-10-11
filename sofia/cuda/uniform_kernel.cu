#include <stdlib.h>
#include <stdio.h>
#include <string.h>

extern "C" {
#include "uniform_kernel.h"
}

#define BLOCKSIZE 32


#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
   if (code != cudaSuccess) {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}



__global__ void uniform_filter_1d_gpu_kernel(double *d_in_cube, double *d_out_cube, size_t allocate_px, int size1, int size2, int stride, int cube_z, int cube_y, int cube_x, int kz) {

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int x = blockDim.x * blockIdx.x + tx;
    int y = blockDim.y * blockIdx.y + ty;

    if (x < cube_x && y < cube_y) {

        // calculate initial average
        double tmp = 0.0;
        int start_idx = y * cube_x + x;;
        int end_idx = start_idx + cube_z * stride;

        // initialize tmp
        for (int i = start_idx; i <= start_idx + (size2 * stride); i += stride) {
            tmp += (i >= end_idx) ? 0.0 : d_in_cube[i];
        }
        tmp /= (double) kz;
        d_out_cube[start_idx] = tmp;

        int lo_idx, hi_idx;
        double lo_val, hi_val;
        for (int cube_idx = start_idx + stride; cube_idx < end_idx; cube_idx += stride) {
            lo_idx = cube_idx - size1 * stride;
            hi_idx = cube_idx + size2 * stride;
            lo_val = (lo_idx < start_idx) ? 0.0 : d_in_cube[lo_idx];
            hi_val = (hi_idx >= end_idx)  ? 0.0 : d_in_cube[hi_idx];
            tmp += hi_val / (double) kz;
            d_out_cube[cube_idx] = tmp;
            // remove lower value for next iteration
            tmp -= lo_val / (double) kz;
        }

    }

    __syncthreads();

}



int get_available_out_bytes_uni() {
    size_t free_byte, total_byte;
    gpuErrchk( cudaMemGetInfo(&free_byte, &total_byte) );
    double free_db = (double)free_byte;
    return (int) (free_db - 1024.0 * 1024.0 * 15) / 2.0; // shave off 15MB so we don't crash by not being able to allocate the last bit of memory on the device
}


void uniform_filter_1d_GPU(double *in, double *out, int cube_z, int cube_y, int cube_x, int kz) {

    // initialize gaussian filter weights
    int size1 = kz / 2;
    int size2 = kz - size1 - 1;
    if (size1 == 0 && size2 <= 0) {
        return;
    }

    size_t stride = cube_x * cube_y;

    // initialize size of cube to filter
    double *h_in = in;
    size_t plane_px = cube_z * cube_x;
    size_t total_cube_px = plane_px * cube_y;
    long total_cube_bytes = total_cube_px * sizeof(double);

    // determine how much memory is available for allocation on the device
    // and calculate how much z_height of the cube can fit into it
    size_t out_bytes = get_available_out_bytes_uni();
    size_t out_px = out_bytes / sizeof(double);
    size_t y_depth;
    if (total_cube_px > out_px) {
        y_depth = (out_px - out_px % plane_px) / plane_px;
    } else {
        y_depth = total_cube_px / plane_px;
    }
    size_t allocate_px = y_depth * plane_px;
    size_t allocate_bytes = allocate_px * sizeof(double);

    // memory allocation
    double *d_in  = NULL;
    double *d_out = NULL;
    gpuErrchk( cudaMalloc(&d_in,  allocate_bytes) );
    gpuErrchk( cudaMalloc(&d_out, allocate_bytes) );

    gpuErrchk( cudaHostRegister(in, total_cube_bytes, 0) );

    // process the cube in parts on the GPU
    size_t offset = 0;
    for (; offset < total_cube_px; offset += allocate_px) {

        if (offset + allocate_px > total_cube_px) {
            y_depth = (total_cube_px - offset) / plane_px;
            allocate_px = y_depth * plane_px;
            allocate_bytes = allocate_px * sizeof(double);
        }

        gpuErrchk( cudaMemcpy(d_in, &h_in[offset], allocate_bytes, cudaMemcpyHostToDevice) );
        dim3 dimBlock = dim3(BLOCKSIZE, BLOCKSIZE, 1);
        dim3 dimGrid = dim3(
            ceil( ((int) cube_x)  / (float) dimBlock.x),
            ceil( ((int) y_depth) / (float) dimBlock.y),
            1
        );
        uniform_filter_1d_gpu_kernel<<<dimGrid, dimBlock>>>(d_in, d_out, allocate_px, size1, size2, stride, cube_z, cube_y, cube_x, kz);
        gpuErrchk( cudaPeekAtLastError() );
        gpuErrchk( cudaMemcpy(&h_in[offset], d_out, allocate_bytes, cudaMemcpyDeviceToHost) );

    }

    // clean up
    cudaHostUnregister(in);
    cudaFree(d_in);
    cudaFree(d_out);

}

