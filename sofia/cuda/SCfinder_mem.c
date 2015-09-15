#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <sys/time.h>
#include <omp.h>

double get_time(struct timeval tv1, struct timeval tv2) {
    double delta = ((tv2.tv_sec - tv1.tv_sec) * 1000000u + tv2.tv_usec - tv1.tv_usec) / 1.e6;
    return delta;
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

void convolve_1d(float *in_cube, float *out_cube, float *weights, int cube_idx, size_t lw, int offset_multiplier, int min_clip, int max_clip) {

    float sum = weights[lw] * in_cube[cube_idx];

    for (size_t i = 1; i < lw + 1; i++) {
        float lo_val = 0.0;
        float hi_val = 0.0;
        float weight = weights[lw + i];
        int cube_offset = i * offset_multiplier;

        int idx_lo = cube_idx - cube_offset;
        int idx_hi = cube_idx + cube_offset;

        if (idx_lo >= min_clip) {
            lo_val = in_cube[idx_lo];
        }
        if (idx_hi < max_clip) {
            hi_val = in_cube[idx_hi];
        }

        sum += weight * (lo_val + hi_val);
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

    size_t stride = cube_x * cube_y;

    struct timeval tv1, tv2;
    // correlate weights with the cube
    if (!switch_xy) {

        // clock_t start = clock();
        gettimeofday(&tv1, NULL);

        #pragma omp parallel for
        for (size_t z = 0; z < cube_z; z++) {
            for (size_t y = 0; y < cube_y; y++) {
                for (size_t x = 0; x < cube_x; x++) {
                    size_t cube_idx = (z * stride) + (y * cube_x) + x;
                    size_t max_clip = cube_idx - x + cube_x;
                    size_t min_clip = max_clip - cube_x;
                    convolve_1d(in_cube, out_cube, weights, cube_idx, lw, 1, min_clip, max_clip);
                }
            }
        }

        gettimeofday(&tv2, NULL);
        printf("contiguous took: %f\n", get_time(tv1, tv2));

    } else {

        // clock_t start = clock();
        gettimeofday(&tv1, NULL);

        #pragma omp parallel for
        for (size_t z = 0; z < cube_z; z++) {
            for (size_t x = 0; x < cube_x; x++) {
                for (size_t y = 0; y < cube_y; y++) {
                    size_t cube_idx = (z * stride) + (y * cube_x) + x;
                    size_t max_clip = cube_idx - (y * cube_x) + stride;
                    size_t min_clip = max_clip - stride;
                    convolve_1d(in_cube, out_cube, weights, cube_idx, lw, cube_x, min_clip, max_clip);
                }
            }
        }

        gettimeofday(&tv2, NULL);
        printf("contiguous took: %f\n", get_time(tv1, tv2));

    }

    free(weights);
}


void gaussian_filter(float *in_cube, float *out_cube, size_t cube_z, size_t cube_y, size_t cube_x, size_t kz, size_t ky, size_t kx) {

    // we recycle the cubes so as to avoid manually copying data across.
    // here in_cube ->(gaussianY)-> out_cube ->(gaussianX)-> in_cube
    if (ky > 0) {
        gaussian_filter_1d(in_cube, out_cube, cube_z, cube_y, cube_x, ky, 1);
    }
    if (kx > 0) {
        gaussian_filter_1d(out_cube, in_cube, cube_z, cube_y, cube_x, kx, 0);
    }

}

void uniform_filter_1d(float *in_cube, float *out_cube, size_t cube_z, size_t cube_y, size_t cube_x, size_t kz) {

    int size1 = kz / 2;
    int size2 = kz - size1 - 1;
    if (size1 == 0 && size2 <= 0) {
        return;
    }

    size_t stride = cube_x * cube_y;
    float tmp;

    #pragma omp parallel for
    for (size_t x = 0; x < cube_x; x++) {
        for (size_t y = 0; y < cube_y; y++) {

            tmp = 0.0;
            int start_idx = (y * cube_x) + x;
            int end_idx = start_idx + cube_z * stride;
            // initialize tmp
            for (int i = start_idx; i <= start_idx + (size2 * stride); i += stride) {
                tmp += (i >= end_idx) ? 0.0 : in_cube[i];
            }
            tmp /= (float) kz;
            out_cube[start_idx] = tmp;

            for (size_t z = start_idx + stride; z < end_idx; z += stride) {
                int lo_idx = z - size1 * stride;
                int hi_idx = z + size2 * stride;
                float lo_val = (lo_idx < start_idx) ? 0.0 : in_cube[lo_idx];
                float hi_val = (hi_idx >= end_idx)  ? 0.0 : in_cube[hi_idx];
                tmp += hi_val / (float) kz;
                out_cube[z] = tmp;
                // remove the lower value for the next iteration
                tmp -= lo_val / (float) kz;
            }
        }
    }

}

void uniform_filter_1d_multi(float *in_cube, float *out_cube, size_t cube_z, size_t cube_y, size_t cube_x, size_t kz) {

    int size1 = kz / 2;
    int size2 = kz - size1 - 1;
    if (size1 == 0 && size2 <= 0) {
        return;
    }

    size_t stride = cube_x * cube_y;

    for (int x = 0; x < cube_x; x++) {
        for (int y = 0; y < cube_y; y++) {
            for (int z = 0; z < cube_z; z++) {

                float avg = 0;

                for (int dz = z-size1; dz <= z+size2; dz++) {
                    int z_idx = MAX(0, dz);
                    z_idx = MIN(cube_z - 1, dz);
                    int index = (z_idx * stride) + (y * cube_x) + x;
                    avg += in_cube[index];
                }

                avg /= size1 + size2 + 1;
                out_cube[(z * stride) + (y * cube_x) + x] = avg;

            }
        }
    }
}

// void SCfinder_mem(float *in_cube, size_t cube_z, size_t cube_y, size_t cube_x, int *kernels, size_t kern_size) {
void SCfinder_mem(float *in_cube, size_t cube_z, size_t cube_y, size_t cube_x, int *kernel) {

    float* out_cube = (float *) malloc(sizeof(float) * cube_x * cube_y * cube_z);

    size_t kx = kernel[0];
    size_t ky = kernel[1];
    size_t kz = kernel[2];
    size_t kt = kernel[3];

    struct timeval tv1, tv2;

    if (kx + ky > 0) {
        gettimeofday(&tv1, NULL);
        gaussian_filter(in_cube, out_cube, cube_z, cube_y, cube_x, 0, ky, kx);
        gettimeofday(&tv2, NULL);
        printf("time spent on gaussian filter: %f\n", get_time(tv1, tv2));
    }
    if (kz > 0) {
        gettimeofday(&tv1, NULL);
        uniform_filter_1d(in_cube, out_cube, cube_z, cube_y, cube_x, kz);
        gettimeofday(&tv2, NULL);
        printf("time spent on uniform filter: %f\n", get_time(tv1, tv2));
    }

    free(out_cube);

}
