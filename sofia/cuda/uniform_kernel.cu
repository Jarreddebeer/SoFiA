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



__global__ void uniform_filter_1d_gpu_kernel(double *d_in_cube, double *d_out_cube, int big_stride, int pad1_px, int pad2_px, int stride, int cube_y, int cube_x, int kz) {

    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x < cube_x && y < cube_y) {

        // calculate initial average
        double tmp = 0.0;
        int start_idx = pad1_px + y * cube_x + x;
        int end_idx = start_idx + big_stride;

        int lo_idx = start_idx - pad1_px;
        int hi_idx = start_idx + pad2_px;

        double lo_val, hi_val;


        // initialize tmp
        for (int i = lo_idx; i <= hi_idx; i += stride) {
            tmp += d_in_cube[i];
        }
        tmp /= (double) kz;
        d_out_cube[start_idx - pad1_px] = tmp;

        /*
        if (x == 319 && y == 319 && big_stride < 30000000) {
            printf("setting start_idx of %d (z level %d) to %f\n", start_idx, start_idx / stride, tmp);
        }
        */

        tmp -= d_in_cube[lo_idx] / (double) kz;
        lo_idx += stride;
        start_idx += stride;
        hi_idx += stride;

        for (int cube_idx = start_idx; cube_idx < end_idx; cube_idx += stride, lo_idx += stride, hi_idx += stride) {
            lo_val = d_in_cube[lo_idx];
            hi_val = d_in_cube[hi_idx];
            tmp += hi_val / (double) kz;
            d_out_cube[cube_idx - pad1_px] = tmp;

            /*
            if (x == 319 && y == 319 && big_stride < 30000000) {
                printf("lo_val: %f, tmp: %f, hi_val: %f z level %d\n", lo_val, tmp, hi_val, cube_idx / stride);
            }
            */

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
    // return 48947200 * 8;
    // return 48230400 * sizeof(double);
    // return (int) (1000000 * sizeof(double));
}


void uniform_filter_1d_GPU(double *in, double *out, int cube_z, int cube_y, int cube_x, int kz) {

    if (cube_z == 0 && cube_y == 0 && cube_x == 0) return;

    gpuErrchk( cudaDeviceReset() );

    // initialize gaussian filter weights
    int size1 = kz / 2;
    int size2 = kz - size1 - 1;
    if (size1 == 0 && size2 <= 0) {
        return;
    }

    int stride = cube_x * cube_y;

    /*
    for (int i = 0; i < cube_z; i++) {
        printf("h_in[%d] = %f\n", i, in[i * stride + 319 * cube_x + 319]);
    }
    */

    // initialize size of cube to filter
    double *h_in = in;
    int plane_px = cube_y * cube_x;
    int total_cube_px = plane_px * cube_z;
    int total_cube_bytes = total_cube_px * sizeof(double);

    // initialize padded data
    int pad1_px = size1 * plane_px;
    int pad2_px = size2 * plane_px;
    int pad_px = pad1_px + pad2_px;
    int pad1_bytes = pad1_px * sizeof(double);
    int pad2_bytes = pad2_px * sizeof(double);
    double* pad1_next = (double*) malloc(pad1_bytes);

    // determine how much memory is available for allocation on the device
    // and calculate how much z_height of the cube can fit into it
    int available_bytes = get_available_out_bytes_uni();
    int available_px = available_bytes / sizeof(double);
    int allocate_px;
    // case 1: there are more pixels than can be allocated to device memory
    if (available_px - pad_px < total_cube_px) {
        allocate_px = (available_px - available_px % plane_px);
    } else {
        allocate_px = total_cube_px + pad_px;
    }
    // size_t allocate_px = z_depth * plane_px;
    // printf("allocating px, allocate_px: %d (z level of: %d)\n", allocate_px, allocate_px / stride);
    int allocate_bytes = allocate_px * sizeof(double);

    // info to process the cube in chunks
    int big_index = -pad1_px;
    int big_stride = allocate_px - pad_px;
    // printf("setting big_stride to %d (z level of: %d)\n", big_stride, big_stride / stride);
    int big_stride_bytes = big_stride * sizeof(double);

    // memory allocation
    double *d_in  = NULL;
    double *d_out = NULL;

    gpuErrchk( cudaMalloc(&d_in,  allocate_bytes) );
    gpuErrchk( cudaMalloc(&d_out, big_stride_bytes) );

    gpuErrchk( cudaHostRegister(h_in, total_cube_bytes, 0) );

    int has_next_run = 1;

    for (; big_index + pad1_px < total_cube_px; big_index += big_stride) {

        // blit d_in with zeros
        gpuErrchk( cudaMemset(d_in, 0, allocate_bytes) );
        // printf("Iteration. big_index is %d (z level of: %d), allocate_px is %d (z level of: %d)\n", big_index, big_index / stride, allocate_px, allocate_px / stride);

        gpuErrchk( cudaDeviceSynchronize() );

        // case 1: end of cube is within our big_stride, so clip big_stride
        if (big_index + pad1_px + big_stride > total_cube_px) {
            // printf("CASE 1!!!\n");
            allocate_px = total_cube_px - big_index;
            // printf("allocate_px: %d (z level of: %d)\n", allocate_px, allocate_px / stride);
            allocate_bytes = allocate_px * sizeof(double);
            big_stride = allocate_px - pad1_px;
            // printf("big_stride: %d (z level of: %d)\n", big_stride, big_stride / stride);
            big_stride_bytes = big_stride * sizeof(double);
            pad2_bytes = 0;
            has_next_run = 0;
        }

        // case 2: end of cube is within our pad2_px region, so clip pad2_px
        else if (big_index + allocate_px > total_cube_px) {
            // printf("CASE 2\n");
            // printf("big_index: %d, big_stride: %d, total_cube_px: %d\n", big_index, big_stride, total_cube_px);
            pad2_px -= (big_index + allocate_px) - total_cube_px;
            pad2_bytes = pad2_px * sizeof(double);
        }

        // copy pad1, then (data & pad2), to the device

        // we can ignore copying from pad1_next on the first iteration because
        // they are all zeros and we initialize with cudaMemset to 0 above.
        if (big_index >= 0) {
            gpuErrchk( cudaMemcpy(d_in, pad1_next, pad1_bytes, cudaMemcpyHostToDevice));
        }

        // populate the device with big_stride and pad2. big_stride and pad2 will have been set to zero or clipped in either case1 or case2 above.
        gpuErrchk( cudaMemcpy(&d_in[pad1_px], &h_in[big_index + pad1_px], big_stride_bytes + pad2_bytes, cudaMemcpyHostToDevice) );

        // cache pad1 data for the next run
        if (has_next_run) {
            memcpy(pad1_next, &h_in[big_index + big_stride], pad1_bytes);
        }

        dim3 dimBlock = dim3(BLOCKSIZE, BLOCKSIZE, 1);
        dim3 dimGrid = dim3(
            ceil( ((int) cube_x) / (float) dimBlock.x),
            ceil( ((int) cube_y) / (float) dimBlock.y),
            1
        );
        // printf("start_idx: %d (z level of %d)\n", pad1_px, pad1_px / stride);
        // printf("end_idx: %d (z level of %d)\n", pad1_px + big_stride, (pad1_px + big_stride) / stride);
        uniform_filter_1d_gpu_kernel<<<dimGrid, dimBlock>>>(d_in, d_out, big_stride, pad1_px, size2 * plane_px, stride, cube_y, cube_x, kz);
        gpuErrchk( cudaPeekAtLastError() );
        gpuErrchk( cudaDeviceSynchronize() );

        // copy back to host data
        // printf("copying d_out to h_in with big_stride %d (z level of %d)\n", big_stride, big_stride / stride);
        gpuErrchk( cudaMemcpy(&h_in[big_index + pad1_px], d_out, big_stride_bytes, cudaMemcpyDeviceToHost) );

    }

    /*
    double* d_in_copy = (double*) malloc(allocate_bytes + (1*320*320) * sizeof(double));
    double* d_out_copy = (double*) malloc(big_stride_bytes);
    cudaMemcpy(d_in_copy, d_in, allocate_bytes + (1*320*320) * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(d_out_copy, d_out, big_stride_bytes, cudaMemcpyDeviceToHost);

    for (int i = 0; i < 6; i++) {
        printf("d_in_copy[%d] = %f\n", i, d_in_copy[i * stride + 319 * cube_x + 319]);
    }

    for (int i = 0; i < big_stride / stride; i++) {
        printf("d_out[%d] = %f\n", i, d_out_copy[i * stride + 319 * cube_x + 319]);
    }

    free(d_in_copy);
    free(d_out_copy);
    */

    // clean up
    cudaHostUnregister(in);
    cudaFree(d_in);
    cudaFree(d_out);
    free(pad1_next);

}

