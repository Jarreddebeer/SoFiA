#include <stdlib.h>
#include <stdio.h>
#include <string.h>

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





__global__ void gaussian_filter_row_gpu(double *d_in_cube, double *d_out_cube, int cube_x, int cube_y, int kernel_r) {

    int x, y, tx, ty;
    __shared__ double subcube[blockDim.y][blockDim.x + 2 * kernel_r];

    tx = threadIdx.x;
    ty = threadIdx.y;
    x = blockDim.x * blockIdx.x + tx;
    y = blockDim.y * blockIdx.y + ty;

    // copy pixel into shared memory
    subcube[ty][tx + kernel_r] = d_in_cube[y * cube_x + x];
    // copy window into shared memory
    for (int i = 1; i <= kernel_r; i++) {
        if (tx == 0) { // left boundary
            if (x == 0) { // cube hard boundary, pad with zero
                subcube[ty][tx + kernel_r - i] = 0; // TODO: pad the cube beforehand rather
            } else { // soft boundary, pad from cube pixels
                subcube[ty][tx + kernel_r - i] = d_in_cube[y * cube_x + x - i];
            }
        } else if (tx == blockDim.x - 1) { // right boundary
            if (x == cube_x - 1) { // cube hard boundary, pad with zero
                subcube[ty][tx + kernel_r + i] = 0;
            } else { // soft boundary, pad from cube pixels
                subcube[ty][tx + kernel_r + i] = d_in_cube[y * cube_x + x + i];
            }
        }
    }

    __syncthreads();

    // perform gaussian filter
    double sum = d_kernel[0] * subcube[ty][tx + kernel_r];
    for (int k = 1; k < 1 + kernel_r; k++) {
        sum += d_kernel[k] * (subcube[ty][tx + kernel_r - k] + subcube[ty][tx + kernel_r + k]);
    }

    d_out_cube[y * cube_x + x] = sum;

    __syncthreads();

}


void gaussian_filter_GPU(double *h_in_cube, double *h_out_cube, size_t cube_z, size_t cube_y, size_t cube_x, size_t ky, size_t kx) {

    int cube_size = cube_z * cube_y * cube_x * sizeof(double)
    int data_w = (int) cube_x;
    int data_h = (int) cube_y;
    int data_size = data_w * data_h * sizeof(double);

    int row_stdev = ((double) kx) / 2.355;
    int row_kernel_w = (int) (row_stdev * 4 + 0.5 ) + 1; // see gaussian_filter_1d

    int row_kernel_size = row_kernel_w * sizeof(double);

    double *d_in_cube = NULL;
    double *d_out_cube = NULL;

    // generate the gaussian weights
    __device__ __constant__ double d_kernel[kernel_w];
    double *h_row_kernel    = (double *) malloc(row_kernel_size);

    float row_kernel_sum = 0;
    // generate row weights
    int i;
    for (i = 0; i < row_kernel_w; i++) {
        h_row_kernel[i] = exp(-0.5 * ((double) i * i) / (row_stdev * row_stdev));
        row_kernel_sum += h_row_kernel[i];
    }
    for (i = 0; i < row_kernel_w; i++) {
        h_kernel[i] /= row_kernel_sum;
    }
    //

    cudaMalloc(&d_in_cube, cube_size);
    cudaMalloc(&d_out_cube, cube_size);

    cudaMemcpy(d_in_cube, h_in_cube, cube_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_out_cube, h_out_cube, cube_size, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(d_kernel, h_kernel, row_kernel_size);

    dim3 dimBlock = dim3(BLOCKSIZE, BLOCKSIZE, 1);
    dim3 dimGrid = dim3(
        ceil( ((int)cube_x) / (float) dimBlock.x),
        ceil( ((int)cube_y) / (float) dimBlock.Y),
        1
    );

    if (kx > 0) {
        gaussian_filter_col_gpu<<<dimGrid, dimBlock>>>(d_in_cube, d_out_cube, cube_x, cube_y, row_kernel_w - 1);

        gpuErrchk( cudaPeekAtLastError() );
        gpuErrchk( cudaDeviceSynchronize() );

        cudaMemcpy(h_out_cube, d_out_cube, cube_size, cudaMemcpyDeviceToHost);
    }

}

