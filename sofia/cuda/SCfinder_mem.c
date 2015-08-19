#include <stdlib.h>
#include <stdio.h>
#include <math.h>

// gcc -shared -o libhello.so -fPIC hello.c

void copy3d(float *to, float *from, size_t cube_z, size_t cube_y, size_t cube_x) {
    for (size_t z = 0; z < cube_z; z++) {
        for (size_t y = 0; y < cube_y; y++) {
            for (size_t x = 0; x < cube_x; x++) {
                size_t cube_idx = (z * cube_y * cube_x) + (y * cube_x) + x;
                to[cube_idx] = from[cube_idx];
            }
        }
    }
}

void convolve_1d(float *in_cube, float *out_cube, float *weights, int cube_idx, size_t lw, int offset_multiplier, int min_clip, int max_clip) {
    float sum = weights[lw] * in_cube[cube_idx];
    if (0) {
        printf("Processing position: %d\n", cube_idx);
        printf("Max and Min clips: %d, %d\n", (int)min_clip, (int)max_clip);
        printf("Sums are: ");
        printf("%f (%d) ", in_cube[cube_idx], cube_idx);
    }
    for (size_t i = 1; i < lw + 1; i++) {
        float weight = weights[lw + i];
        int cube_offset = i * offset_multiplier;

        // we assume the window size (lw) will never be more than one full cube dimension
        int idx_hi = cube_idx + cube_offset;
        if (idx_hi >= max_clip) {
            int delta = idx_hi - (max_clip - offset_multiplier);
            idx_hi = max_clip - delta;
        }

        int idx_lo = cube_idx - cube_offset;
        if (idx_lo < min_clip) {
            int delta = (min_clip - offset_multiplier) - idx_lo;
            idx_lo = min_clip + delta;
        }
        if (0) {
            printf("%f (%d)", in_cube[idx_lo], idx_lo);
            printf("%f (%d)", in_cube[idx_hi], idx_hi);
        }
        sum += weight * in_cube[idx_lo];
        sum += weight * in_cube[idx_hi];
    }
    if (0) {
        printf("\n");
        printf("Total sum: %f\n", sum);
    }
    out_cube[cube_idx] = sum;
}


void gaussian_filter_1d(float *in_cube, float *out_cube, size_t cube_z, size_t cube_y, size_t cube_x, size_t ks, int switch_xy) {

    int truncate = 4;

    float sd = ((float) ks) / 2.355;
    int lw = (int) (sd * truncate + 0.5);

    float* weights = (float *) malloc(sizeof(float) * (2 * lw + 1));
    weights[lw] = 1.0;
    float sum = 1.0;

    // generate the weights

    for (size_t i = 1; i < lw + 1; i++) {
        float tmp = exp(-0.5 * ((float) i * i) / (sd * sd));
        weights[lw + i] = tmp;
        weights[lw - i] = tmp;
        sum += 2.0 * tmp;
    }

    for (size_t i = 0; i < 2 * lw + 1; i++) {
        weights[i] /= sum;
    }

    if (0) {
        printf("sd: %f\n", sd);
        printf("lw: %d\n", lw);
        printf("sum: %f\n", sum);
        printf("weights: ");

        for (size_t i = 0; i < 2 * lw + 1; i++) {
            printf("%f ", weights[i]);
        }
        printf("\n");
    }

    // correlate weights with the cube
    if (!switch_xy) {
        // printf("cube_x: %d\n", (int)cube_x);

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
        // printf("cube_x: %d\n", (int)cube_x);
        // printf("cube_y: %d\n", (int)cube_y);

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

void gaussian_filter(float *in_cube, size_t cube_z, size_t cube_y, size_t cube_x, size_t kz, size_t ky, size_t kx) {

    float* out_cube = (float *) malloc(sizeof(float) * cube_x * cube_y * cube_z);

    gaussian_filter_1d(in_cube, out_cube, cube_z, cube_y, cube_x, ky, 1);
    copy3d(in_cube, out_cube, cube_z, cube_y, cube_x);
    gaussian_filter_1d(in_cube, out_cube, cube_z, cube_y, cube_x, kx, 0);
    copy3d(in_cube, out_cube, cube_z, cube_y, cube_x);

    free(out_cube);
}

void uniform_filter_1d(float *in_cube, size_t cube_z, size_t cube_y, size_t cube_x, size_t kz) {

    float* out_cube = (float *) malloc(sizeof(float) * cube_x * cube_y * cube_z);

    int tmp = 0.0;

    for (size_t x = 0; x < cube_x; x++) {
        for (size_t y = 0; y < cube_y; y++) {

            // window reflected average
            for (size_t z = 0; z < kz; z++) {
                size_t cube_idx = (z * cube_y * cube_x) + (y * cube_x) + x;
                tmp += in_cube[cube_idx];
            }
            tmp /= (float) kz;

            for (size_t z = 0; z < cube_z; z++) {
                size_t cube_idx = (z * cube_y * cube_x) + (y * cube_x) + x;
                size_t lo_idx;
                // handle mirrored case
                if (z < kz) {
                    lo_idx = ((kz - 1 - z) * cube_y * cube_x) + (y * cube_x) + x;
                } else {
                    lo_idx = cube_idx - (kz * cube_y * cube_x);
                }
                tmp += (in_cube[cube_idx] - in_cube[lo_idx]) / (float) kz;
                out_cube[cube_idx] = tmp;
            }
        }
    }

    copy3d(in_cube, out_cube, cube_z, cube_y, cube_x);
    free(out_cube);
}

void SCfinder_mem(float *in_cube, size_t cube_z, size_t cube_y, size_t cube_x, int *kernels, size_t kern_size) {

    size_t k;
    for (k = 0; k < kern_size; k++) {

        size_t k_idx = k * 4;
        int kx = kernels[k_idx + 0];
        int ky = kernels[k_idx + 1];
        int kz = kernels[k_idx + 2];
        int kt = kernels[k_idx + 3];

        gaussian_filter(in_cube, cube_z, cube_y, cube_x, kz, ky, kx);
        uniform_filter_1d(in_cube, cube_z, cube_y, cube_x, kz);

    }

}
