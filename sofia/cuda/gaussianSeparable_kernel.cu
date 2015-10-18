#include <stdlib.h>
#include <stdio.h>
#include <string.h>

extern "C" {
#include "gaussianSeparable_kernel.h"
}

#define BLOCKSIZE 16
#define MAX_LW 257 // CONSTRAINT: max size of lw (window) is 256
__device__ __constant__ float d_weights[MAX_LW];


#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
   if (code != cudaSuccess) {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}


void copy3d(float *to, float *from, size_t cube_z, size_t cube_y, size_t cube_x) {
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


void print_memory_use() {
    size_t free_byte ;
    size_t total_byte ;
    cudaError_t cuda_status = cudaMemGetInfo( &free_byte, &total_byte ) ;
    if ( cudaSuccess != cuda_status ){
        printf("Error: cudaMemGetInfo fails, %s \n", cudaGetErrorString(cuda_status) );
        exit(1);
    }
    double free_db = (double)free_byte ;
    double total_db = (double)total_byte ;
    double used_db = total_db - free_db ;
    printf("GPU memory usage: used = %f, free = %f MB, total = %f MB\n",
    used_db/1024.0/1024.0, free_db/1024.0/1024.0, total_db/1024.0/1024.0);
}




extern "C" double* init_out_cube(size_t cube_z, size_t cube_y, size_t cube_x) {
    double* out_cube;
    gpuErrchk( cudaMallocHost(&out_cube, sizeof(double) * cube_x * cube_y * cube_z) );
    return out_cube;
}







__device__ float convolve_1d_gpu_kernel_optimised(float *subcube, int sidx, int lw) {

    float sum = d_weights[lw] * subcube[sidx];
    float lo_val, hi_val, weight;

    for (int i = 1; i <= lw; i++) {
        weight = d_weights[lw + i];
        lo_val = subcube[sidx - i];
        hi_val = subcube[sidx + i];
        sum += weight * (lo_val + hi_val);
    }

    return sum;
}







__global__ void gaussian_filter_1d_gpu_kernel_optimised(float *d_in_cube, float *d_out_cube, size_t allocate_px, int lw, int stride, int cube_y, int cube_x, int stream_offset, int ppt) {

    int tx, ty, sx, sy, x, y, sidx, sstride, cube_idx, t, ct, end;
    extern __shared__ float subcube[];

    tx = threadIdx.x;
    ty = threadIdx.y;
    sstride = ppt * BLOCKSIZE + 2 * lw;

    // if stride == cube_x then we are filtering along y. so load the data into shared memory contiguously for processing
    if (stride == cube_x) {
        t = ty;
        end = cube_y;
        sx = tx;
        sy = lw + ty * ppt;
        sidx = sx * sstride + sy;
        x = blockDim.x * blockIdx.x + tx;
        y = (blockDim.y * blockIdx.y + ty) * ppt;
        ct = y;
    } else {
        t = tx;
        end = cube_x;
        sx = lw + tx * ppt;
        sy = ty;
        sidx = sy * sstride + sx;
        x = (blockDim.x * blockIdx.x + tx) * ppt;
        y = blockDim.y * blockIdx.y + ty;
        ct = x;
    }

    cube_idx = stream_offset + (blockIdx.z * cube_x * cube_y) + (y * cube_x) + x;

    // copy pixel into subcube shared memory
    ////////////////////////////////////////

    if (x >= cube_x || y >= cube_y) {
        for (int i = 0; i < ppt; i++) {
            subcube[sidx + i] = 0;
        }
    } else {
        for (int i = 0; i < ppt; i++) {
            subcube[sidx + i] = d_in_cube[cube_idx + i * stride];
        }
    }

    // copy window into shared memory
    ////////////////////////////////////

    // padding
    if (t == 0) { // left boundary
        if (ct == 0) { // cube hard boundary, pad with zero
            for (int i = 1; i <= lw; i++) {
                subcube[sidx - i] = 0;
            }
        } else { // soft boundary, pad from cube pixels
            for (int i = 1; i <= lw; i++) {
                subcube[sidx - i] = d_in_cube[cube_idx - i * stride];
            }
        }

    } else if (t == BLOCKSIZE - 1) { // right boundary
        if ( (ct+ppt-1) == end - 1) { // cube hard boundary, pad with zero
            for (int i = 1; i <= lw; i++) {
                subcube[sidx + ppt-1 + i] = 0;
            }
        } else { // soft boundary, pad from cube pixels
            for (int i = 1; i <= lw; i++) {
                if ( (ct+ppt-1) + i < end) { // the window column might be thicker than the remaining columns in the cube
                    subcube[sidx + ppt-1 + i] = d_in_cube[cube_idx + (ppt-1 + i) * stride];
                } else {
                    subcube[sidx + ppt-1 + i] = 0;
                }
            }
        }
    }


    /*
    __syncthreads();

    if (x == 0 && y == 0) {
        printf("end: %d\n", end);
        printf("----------\n");
        for (int i = 0; i < BLOCKSIZE; i++) {
            for (int j = 0; j < BLOCKSIZE * ppt + 2 * lw; j++) {
                printf("%f ", subcube[i * (BLOCKSIZE * ppt + 2 * lw) + j]);
            }
            printf("\n");
        }
        printf("----------\n");
        printf("\n");
    }
    */

    __syncthreads();

    // perform convolution
    //////////////////////

    if (x < cube_x && y < cube_y && cube_idx < allocate_px) {
        for (int i = 0; i < ppt; i++) {
            d_out_cube[cube_idx + i * stride] = convolve_1d_gpu_kernel_optimised(subcube, sidx + i, lw);
        }
    }

}

__device__ float convolve_1d_gpu_kernel(float *subcube, int sidx, int lw, int sstride) {

    float sum = d_weights[lw] * subcube[sidx];
    float lo_val, hi_val, weight;

    for (int i = 1; i <= lw; i++) {
        weight = d_weights[lw + i];
        lo_val = subcube[sidx - i * sstride];
        hi_val = subcube[sidx + i * sstride];
        sum += weight * (lo_val + hi_val);
    }

    return sum;
}

__global__ void gaussian_filter_1d_gpu_kernel(float *d_in_cube, float *d_out_cube, size_t allocate_px, int lw, int stride, int cube_y, int cube_x) {

    int tx, ty, sx, sy, x, y, sidx, sstride, cube_idx;
    extern __shared__ float subcube[];

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


    __syncthreads();


    // perform convolution
    //////////////////////

    if (x < cube_x && y < cube_y && cube_idx < allocate_px) {
        if (stride == 1) { // need to differentiate between x and y convolution
            d_out_cube[cube_idx] = convolve_1d_gpu_kernel(subcube, sidx, lw, 1);
        } else {
            d_out_cube[cube_idx] = convolve_1d_gpu_kernel(subcube, sidx, lw, sstride);
        }
    }

}

float* generate_weights(float sd, int lw) {
    float *h_weights    = (float *) malloc(sizeof(float) * MAX_LW);
    h_weights[lw] = 1.0;
    float sum = 1.0;
    // generate gaussian row weights
    for (int i = 1; i <= lw; i++) {
        float tmp = exp(-0.5 * ((float) i * i) / (sd * sd));
        h_weights[lw + i] = tmp;
        h_weights[lw - i] = tmp;
        sum += 2.0 * tmp;
    }
    for (int i = 0; i <= 2 * lw; i++) {
        h_weights[i] /= sum;
    }
    return h_weights;
}


int get_available_out_bytes() {
    size_t free_byte, total_byte;
    gpuErrchk( cudaMemGetInfo(&free_byte, &total_byte) );
    double free_db = (double)free_byte;
    return (int) (free_db - 1024.0 * 1024.0 * 15) / 2.0; // shave off 15MB so we don't crash by not being able to allocate the last bit of memory on the device
}

void gaussian_filter_1d(float *in, float *out, int cube_z, int cube_y, int cube_x, int ks, int stride) {

    // initialize gaussian filter weights
    float sd = ((float) ks) / 2.355;
    int lw = (int) (sd * 4 + 0.5);

    int ppt = 4; // pixels per thread
    int sm_bytes_optimised = (BLOCKSIZE) * (ppt * BLOCKSIZE + 2 * lw) * sizeof(float);
    float* h_weights = generate_weights(sd, lw);

    // initialize size of cube to filter
    float *h_in = in;
    size_t plane_px = cube_y * cube_x;
    size_t total_cube_px = plane_px * cube_z;
    long total_cube_bytes = total_cube_px * sizeof(float);

    // determine how much memory is available for allocation on the device
    // and calculate how much z_height of the cube can fit into it
    size_t out_bytes = get_available_out_bytes();
    size_t out_px = out_bytes / sizeof(float);
    size_t z_height;
    if (total_cube_px > out_px) {
        z_height = (out_px - out_px % plane_px) / plane_px;
    } else {
        z_height = total_cube_px / plane_px;
    }
    size_t allocate_px = z_height * plane_px;
    size_t allocate_bytes = allocate_px * sizeof(float);

    // memory allocation
    float *d_in  = NULL;
    float *d_out = NULL;
    gpuErrchk( cudaMalloc(&d_in,  allocate_bytes) );
    gpuErrchk( cudaMalloc(&d_out, allocate_bytes) );

    gpuErrchk( cudaMemcpyToSymbol(d_weights, h_weights, MAX_LW * sizeof(float)) );
    gpuErrchk( cudaHostRegister(in, total_cube_bytes, 0) );

    int num_streams;
    if (z_height > 2) {
        num_streams = 2;
    } else {
        num_streams = 1;
    }
    cudaStream_t* stream = (cudaStream_t*) malloc(num_streams * sizeof(cudaStream_t) );
    for (int i = 0; i < num_streams; i++) {
        cudaStreamCreate(&stream[i]);
    }

    // process the cube in parts on the GPU
    size_t offset = 0;
    for (; offset < total_cube_px; offset += allocate_px) {

        for (int i = 0; i < num_streams; i++) {
            gpuErrchk( cudaStreamCreate(&stream[i]) );
        }

        if (offset + allocate_px > total_cube_px) {
            z_height = (total_cube_px - offset) / plane_px;
            allocate_px = z_height * plane_px;
            allocate_bytes = allocate_px * sizeof(float);
        }

        // copy memory asynchronously h -> d
        int z_height_stream_reset = ceil( z_height / (float)num_streams);
        int z_height_stream = z_height_stream_reset;
        int stream_size = z_height_stream * plane_px;
        int stream_size_bytes = stream_size * sizeof(float);

        for (int i = 0; i < num_streams; i++) {

            // must calculate offset before we clip stream_size on the last iteration
            int stream_offset = i * stream_size;

            if (i == num_streams - 1) {
                z_height_stream = z_height - i * z_height_stream;
                if (z_height_stream == 0) break;
                stream_size = z_height_stream * plane_px;
                stream_size_bytes = stream_size * sizeof(float);
            }

            gpuErrchk( cudaMemcpyAsync(&d_in[stream_offset], &h_in[offset + stream_offset], stream_size_bytes, cudaMemcpyHostToDevice, stream[i]) );
        }

        // execute kernel asynchronously
        z_height_stream = z_height_stream_reset;
        stream_size = z_height_stream * plane_px;
        stream_size_bytes = stream_size * sizeof(float);

        for (int i = 0; i < num_streams; i++) {

            // must calculate offset before we clip stream_size on the last iteration
            int stream_offset = i * stream_size;

            if (i == num_streams - 1) {
                z_height_stream = z_height - i * z_height_stream;
                if (z_height_stream == 0) break;
                stream_size = z_height_stream * plane_px;
                stream_size_bytes = stream_size * sizeof(float);
            }

            dim3 dimBlock = dim3(BLOCKSIZE, BLOCKSIZE, 1);
            dim3 dimGrid;
            if (stride == cube_x) { // filtering Y
                dimGrid = dim3(
                    ceil( ((int) cube_x) / (float) dimBlock.x),
                    ceil( ((int) cube_y) / (float) (dimBlock.y * ppt)),
                    ceil( ((int) z_height_stream) / (float) dimBlock.z)
                );
            } else { // filtering X
                dimGrid = dim3(
                    ceil( ((int) cube_x) / (float) (dimBlock.x * ppt)),
                    ceil( ((int) cube_y) / (float) dimBlock.y),
                    ceil( ((int) z_height_stream) / (float) dimBlock.z)
                );
            }
            // printf("dimGrid.x: %d, dimGrid.y: %d. dimBlock.x: %d, dimBlock.y: %d\n", dimGrid.x, dimGrid.y, dimBlock.x, dimBlock.y);
            gaussian_filter_1d_gpu_kernel_optimised<<<dimGrid, dimBlock, sm_bytes_optimised, stream[i]>>>(d_in, d_out, allocate_px, lw, stride, cube_y, cube_x, stream_offset, ppt);
            gpuErrchk( cudaPeekAtLastError() );
        }


        // copy back memory asynchronously d -> h
        z_height_stream = z_height_stream_reset;
        stream_size = z_height_stream * plane_px;
        stream_size_bytes = stream_size * sizeof(float);

        for (int i = 0; i < num_streams; i++) {

            // must calculate offset before we clip stream_size on the last iteration
            int stream_offset = i * stream_size;

            if (i == num_streams - 1) {
                z_height_stream = z_height - i * z_height_stream;
                if (z_height_stream == 0) break;
                stream_size = z_height_stream * plane_px;
                stream_size_bytes = stream_size * sizeof(float);
            }

            gpuErrchk( cudaMemcpyAsync(&h_in[offset + stream_offset], &d_out[stream_offset], stream_size_bytes, cudaMemcpyDeviceToHost, stream[i]) );
        }

        gpuErrchk( cudaDeviceSynchronize() );

        for (int i = 0; i < num_streams; i++) {
            gpuErrchk( cudaStreamDestroy(stream[i]) );
        }

    }

    // clean up
    cudaHostUnregister(in);
    cudaFree(d_in);
    cudaFree(d_out);
    free(h_weights);

}



void gaussian_filter_GPU(float *in_cube, float *out_cube, int cube_z, int cube_y, int cube_x, int ky, int kx) {

    if (cube_z == 0 && cube_y == 0 && cube_x == 0) return;

    int stride;

    if (ky > 0) {
        stride = cube_x;
        gaussian_filter_1d(in_cube, out_cube, cube_z, cube_y, cube_x, ky, stride);
        // copy3d(in_cube, out_cube, cube_z, cube_y, cube_x);
    }

    if (kx > 0) {
        stride = 1;
        gaussian_filter_1d(in_cube, out_cube, cube_z, cube_y, cube_x, kx, stride);
        // copy3d(in_cube, out_cube, cube_z, cube_y, cube_x);
    }

}

