#pragma once
#define HEADER 1

extern void uniform_filter_1d_GPU(float *in_cube, float *out_cube, int cube_z, int cube_y, int cube_x, int kz);
