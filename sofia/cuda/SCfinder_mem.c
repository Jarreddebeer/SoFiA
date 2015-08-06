#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#define MIN(x, y) (((x) < (y)) ? (x) : (y))

// gcc -shared -o libhello.so -fPIC hello.c

void convolve_1d(float *in_cube, float *out_cube, float *weights, int cube_idx, size_t lw, int offset_multiplier, int min_clip, int max_clip) {
    float sum = weights[lw] * in_cube[cube_idx];
    printf("Processing position: %d\n", cube_idx);
    printf("Max and Min clips: %d, %d\n", (int)min_clip, (int)max_clip);
    printf("Sums are: ");
    printf("%f (%d) ", in_cube[cube_idx], cube_idx);
    for (size_t i = 1; i < lw + 1; i++) {
        float weight = weights[lw + i];
        int cube_offset = i * offset_multiplier;
        int idx_hi = MAX(min_clip,     MIN( cube_idx + cube_offset, max_clip - 1 ));
        int idx_lo = MIN( (max_clip) - 1, MAX( cube_idx - cube_offset, min_clip     ));
        printf("%f (%d)", in_cube[idx_lo], idx_lo);
        printf("%f (%d)", in_cube[idx_hi], idx_hi);
        sum += weight * in_cube[idx_lo];
        sum += weight * in_cube[idx_hi];
    }
    printf("\n");
    printf("Total sum: %f\n", sum);
    out_cube[cube_idx] = sum;
}


void gaussian_filter_1d(float *in_cube, float *out_cube, size_t cube_z, size_t cube_y, size_t cube_x, size_t ks, int switch_xy) {

    int truncate = 1;

    float sd = ((float) ks) / 2.355;
    int lw = (int) (truncate * sd + 0.5);

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

    printf("sd: %f\n", sd);
    printf("lw: %d\n", lw);
    printf("sum: %f\n", sum);
    printf("weights: ");
    for (size_t i = 0; i < 2 * lw + 1; i++) {
        printf("%f ", weights[i]);
    }
    printf("\n");

    // correlate weights with the cube
    if (!switch_xy) {
        printf("cube_x: %d\n", (int)cube_x);

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

void gaussian_filter(float *in_cube, float *out_cube, size_t cube_z, size_t cube_y, size_t cube_x, size_t kz, size_t ky, size_t kx) {
    // filter x, then y
    gaussian_filter_1d(in_cube, out_cube, cube_z, cube_y, cube_x, kx, 0);
    // gaussian_filter_1d(in_cube, out_cube,  cube_z, cube_x, cube_y, ky, 1);
}

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

void SCfinder_mem(float *in_cube, size_t cube_z, size_t cube_y, size_t cube_x, int *kernels, size_t kern_size) {

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

        gaussian_filter(in_cube, out_cube, cube_z, cube_y, cube_x, kz, ky, kx);
        copy3d(in_cube, out_cube, cube_z, cube_y, cube_x);

        free(out_cube);
    }

}
