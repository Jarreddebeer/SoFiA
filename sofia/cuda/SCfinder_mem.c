#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>

#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#define MIN(x, y) (((x) < (y)) ? (x) : (y))

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
        float lo_val = 0.0;
        float hi_val = 0.0;
        float weight = weights[lw + i];
        int cube_offset = i * offset_multiplier;

        // int idx_lo = MAX(min_clip,     MIN( cube_idx + cube_offset, max_clip - 1 ));
        // int idx_hi = MIN(max_clip - 1, MAX( cube_idx - cube_offset, min_clip     ));

        int idx_lo = cube_idx - cube_offset;
        int idx_hi = cube_idx + cube_offset;

        if (idx_lo >= min_clip) {
            lo_val = in_cube[idx_lo];
        }
        if (idx_hi < max_clip) {
            hi_val = in_cube[idx_hi];
        }


        /*
        if (idx_hi >= max_clip) {
            // int delta = idx_hi - (max_clip - offset_multiplier);
            // idx_hi = max_clip - delta;
        }

        if (idx_lo < min_clip) {
            int delta = (min_clip - offset_multiplier) - idx_lo;
            idx_lo = min_clip + delta;
        }
        */

        if (0) {
            printf("%f (%d)", lo_val, idx_lo);
            printf("%f (%d)", hi_val, idx_hi);
        }

        // sum += weight * in_cube[idx_lo];
        // sum += weight * in_cube[idx_hi];
        sum += weight * (lo_val + hi_val);
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

    size_t stride = cube_x * cube_y;

    // correlate weights with the cube
    if (!switch_xy) {
        // printf("cube_x: %d\n", (int)cube_x);

        clock_t start = clock();

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

        clock_t end = clock();
        printf("contiguous took: %d\n", (int) (end - start));

    } else {
        // printf("cube_x: %d\n", (int)cube_x);
        // printf("cube_y: %d\n", (int)cube_y);

        clock_t start = clock();

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

        clock_t end = clock();
        printf("non-contiguous took: %d\n", (int) (end - start));

    }

    free(weights);
}

void gaussian_filter(float *in_cube, float *out_cube, size_t cube_z, size_t cube_y, size_t cube_x, size_t kz, size_t ky, size_t kx) {

    // we recycle the cubes so as to avoid manually copying data across.
    // here in_cube ->(gaussianY)-> out_cube ->(gaussianX)-> in_cube
    if (ky > 0) {
        gaussian_filter_1d(in_cube, out_cube, cube_z, cube_y, cube_x, ky, 1);
        // copy3d(in_cube, out_cube, cube_z, cube_y, cube_x);
    }
    if (kx > 0) {
        gaussian_filter_1d(out_cube, in_cube, cube_z, cube_y, cube_x, kx, 0);
        // copy3d(in_cube, out_cube, cube_z, cube_y, cube_x);
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

    for (size_t x = 0; x < cube_x; x++) {
        for (size_t y = 0; y < cube_y; y++) {

            tmp = 0.0;
            int start_idx = (y * cube_x) + x;
            /*
            if (x == 0 && y == 1) {
                printf("start_idx: %d\n", (int)start_idx);
            }
            */
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
                /*
                if (x == 0 && y == 1) {
                    printf("isLower: %d, lo: %f, lo_idx: %d, hi: %f, hi_idx: %d\n", lo_idx < start_idx, lo_val, (int)lo_idx, hi_val, (int)hi_idx);
                }
                */
                tmp += hi_val / (float) kz;
                out_cube[z] = tmp;
                // remove the lower value for the next iteration
                tmp -= lo_val / (float) kz;
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

    if (kx + ky > 0) {
        clock_t start = clock();
        gaussian_filter(in_cube, out_cube, cube_z, cube_y, cube_x, 0, ky, kx);
        clock_t end = clock();
        printf("time spent on gaussian filter: %d\n", ((int) end - (int) start));
    }
    if (kz > 0) {
        clock_t start = clock();
        uniform_filter_1d(in_cube, out_cube, cube_z, cube_y, cube_x, kz);
        clock_t end = clock();
        printf("time spent on uniform filter: %d\n", ((int) end - (int) start));
    }

    free(out_cube);

}

/*
int main() {

    int cube_x = 360;
    int cube_y = 360;
    int cube_z = 660;
    float* in_cube = (float *) malloc(sizeof(float) * cube_x * cube_y * cube_z);
    int* kernels = (int *) malloc(sizeof(int));

    int pos = 0;
    for (int i = 0; i < cube_x; i++) {
        for (int j = 0; j < cube_y; j++) {
            for (int k = 0; k < cube_z; k++) {
                in_cube[pos] = -0.001;
                pos++;
            }
        }
    }

    SCfinder_mem(in_cube, cube_z, cube_y, cube_x, kernels, 1);

}
*/
