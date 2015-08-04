#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#define MIN(x, y) (((x) < (y)) ? (x) : (y))

// gcc -shared -o libhello.so -fPIC hello.c

void convolve_1d(float *in_cube, float *out_cube, float *weights, int cube_idx, size_t lw, int offset_multiplier, size_t min_clip, size_t max_clip) {
    int sum = 0;
    for (size_t i = 1; i < lw + 1; i++) {
        float weight = weights[lw + i];
        size_t cube_offset = i * offset_multiplier;
        size_t idx = MAX(min_clip, MIN( cube_idx + cube_offset, max_clip - 1 ));
        sum += weight * in_cube[idx];
        sum += weight * in_cube[idx];
    }
    out_cube[cube_idx] = sum;
}


void gaussian_filter_1d(float *in_cube, float *out_cube, size_t cube_x, size_t cube_y, size_t cube_z, size_t ks, int switch_xy) {

    int truncate = 4;

    float sd = ((float) ks) / 2.355;
    int lw = (int) (truncate * sd + 0.5);

    float* weights = (float *) malloc(sizeof(float) * (2 * lw + 1));
    weights[lw] = 1.0;
    float sum = 1.0;

    // generate the weights

    for (size_t i = 1; i < lw + 1; i++) {
        float tmp = (1 / sd * sqrt(2 * M_PI)) * exp(-0.5 * ((float) i * i) / (sd * sd));
        weights[lw + i] = tmp;
        weights[lw - i] = tmp;
        sum += 2.0 * tmp;
    }

    for (size_t i = 0; i < 2 * lw + 1; i++) {
        weights[i] /= sum;
    }

    // correlate weights with the cube
    if (!switch_xy) {

        for (size_t z = 0; z < cube_z; z++) {
            for (size_t y = 0; y < cube_y; y++) {
                for (size_t x = 0; x < cube_x; x++) {
                    size_t cube_idx = (z * cube_y * cube_x) + (y * cube_x) + x;
                    size_t max_clip = cube_idx - x + cube_x;
                    size_t min_clip = max_clip - cube_x;
                    convolve_1d(in_cube, out_cube, weights, cube_idx, lw, 1, min_clip, max_clip);
                }
            }
        }

    } else {

        for (size_t z = 0; z < cube_z; z++) {
            for (size_t x = 0; x < cube_x; x++) {
                for (size_t y = 0; y < cube_y; y++) {
                    size_t cube_idx = (z * cube_y * cube_x) + (y * cube_x) + x;
                    size_t max_clip = cube_idx - (y * cube_x) + (cube_x * cube_y);
                    size_t min_clip = max_clip - (cube_x * cube_y);
                    convolve_1d(in_cube, out_cube, weights, cube_idx, lw, cube_x, min_clip, max_clip);
                }
            }
        }

    }

    free(weights);
}

void gaussian_filter(float *in_cube, float *out_cube, size_t cube_x, size_t cube_y, size_t cube_z, size_t kx, size_t ky, size_t kz) {
    // filter x, then y
    gaussian_filter_1d(in_cube, out_cube, cube_x, cube_y, cube_z, kx, 0);
    gaussian_filter_1d(in_cube, out_cube,  cube_y, cube_x, cube_z, ky, 1);
}

void copy3d(float *to, float *from, size_t cube_x, size_t cube_y, size_t cube_z) {
    for (size_t z = 0; z < cube_z; z++) {
        for (size_t y = 0; y < cube_y; y++) {
            for (size_t x = 0; x < cube_x; x++) {
                size_t cube_idx = (z * cube_y * cube_x) + (y * cube_x) + x;
                to[cube_idx] = from[cube_idx];
            }
        }
    }
}

void SCfinder_mem(float *in_cube, size_t cube_x, size_t cube_y, size_t cube_z, int *kernels, size_t kern_size) {

    size_t k;
    for (k = 0; k < kern_size; k++) {

        float* out_cube = (float *) malloc(sizeof(float) * cube_x * cube_y * cube_z);

        size_t k_idx = k * 4;
        int kx = kernels[k_idx + 0];
        int ky = kernels[k_idx + 1];
        int kz = kernels[k_idx + 2];
        int kt = kernels[k_idx + 3];

        // gaussian filter in x&y
        // uniform filter in z

        gaussian_filter(in_cube, out_cube, cube_x, cube_y, cube_z, kx, ky, kz);
        copy3d(in_cube, out_cube, cube_x, cube_y, cube_z);

        free(out_cube);
    }

}
