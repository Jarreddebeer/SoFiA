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

    int base = size1 * (cube_x * cube_y);

    if (x < cube_x && y < cube_y) {

        // calculate initial average
        double tmp = 0.0;
        int start_idx = base + y * cube_x + x;
        int end_idx = start_idx + cube_z * stride;

        int lo_idx = start_idx - (size1 * stride);
        int hi_idx = start_idx + (size2 * stride);

        double lo_val, hi_val;

        // initialize tmp
        for (int i = lo_idx; i <= hi_idx; i += stride) {
            tmp += d_in_cube[i];
        }
        tmp /= (double) kz;
        d_out_cube[start_idx - base] = tmp;

        lo_idx += stride;
        start_idx += stride;
        hi_idx += stride;

        for (int cube_idx = start_idx; cube_idx < end_idx; cube_idx += stride, lo_idx += stride, hi_idx += stride) {
            lo_val = d_in_cube[lo_idx];
            hi_val = d_in_cube[hi_idx];
            tmp += hi_val / (double) kz;
            d_out_cube[cube_idx - base] = tmp;
            // remove lower value for next iteration
            tmp -= lo_val / (double) kz;
        }

    }

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

    // initialize padded data
    size_t pad1_px = size1 * plane_px;
    size_t pad2_px = size2 * plane_px;
    size_t pad_px = pad1_px + pad2_px;
    size_t pad1_bytes = pad1_px * sizeof(double);
    size_t pad2_bytes = pad2_px * sizeof(double);
    double* pad1_next = malloc(pad1_bytes);

    // determine how much memory is available for allocation on the device
    // and calculate how much z_height of the cube can fit into it
    size_t available_bytes = get_available_out_bytes_uni();
    size_t available_px = available_bytes / sizeof(double);
    size_t z_depth;
    if (total_cube_px > available_px) {
        z_depth = (available_px - available_px % plane_px) / plane_px;
    } else {
        z_depth = total_cube_px / plane_px;
    }
    size_t allocate_px = z_depth * plane_px;
    size_t allocate_bytes = allocate_px * sizeof(double);

    // info to process the cube in chunks
    size_t big_index = -pad1_px;
    size_t big_stride = allocate_px - pad_px;

    size_t out_bytes = big_stride * sizeof(double);

    // memory allocation
    double *d_in  = NULL;
    double *d_out = NULL;
    gpuErrchk( cudaMalloc(&d_in,  allocate_bytes) );
    gpuErrchk( cudaMalloc(&d_out, out_bytes) );

    gpuErrchk( cudaHostRegister(in, total_cube_bytes, 0) );

    for (; big_index < total_cube_px; big_index += big_stride) {

        if (big_index + big_stride > total_cube_px) {
            z_depth = (total_cube_px - big_stride) / plane_px;
            allocate_px = z_depth * plane_px;
            allocate_bytes = allocate_px * sizeof(double);
        }

        // first iteration padding is with zeros
        if (big_index < 0) {
            memset(pad1_next, 0, pad1_bytes);
        }

        // copy pad1, then (data & pad2), to the device
        gpuErrchk( cudaMemcpy(d_in, pad1_next))
        gpuErrchk( cudaMemcpy(&d_in[pad1_px], &h_in[big_index + pad1_px], big_stride + pad2_bytes, cudaMemcpyHostToDevice) );

        // cache pad1 data for the next run
        memcpy(pad1_next, h_in[big_index + big_stride], pad1_bytes);

        dim3 dimBlock = dim3(BLOCKSIZE, BLOCKSIZE, 1);
        dim3 dimGrid = dim3(
            ceil( ((int) cube_x) / (float) dimBlock.x),
            ceil( ((int) cube_y) / (float) dimBlock.y),
            1
        );
        uniform_filter_1d_gpu_kernel<<<dimGrid, dimBlock>>>(d_in, d_out, allocate_px, size1, size2, stride, cube_z, cube_y, cube_x, kz);
        gpuErrchk( cudaPeekAtLastError() );

        // copy back to host data
        gpuErrchk( cudaMemcpy(&h_in[big_index + size1_px], d_out, out_bytes, cudaMemcpyDeviceToHost) );

    }

    // clean up
    cudaHostUnregister(in);
    cudaFree(d_in);
    cudaFree(d_out);

}

