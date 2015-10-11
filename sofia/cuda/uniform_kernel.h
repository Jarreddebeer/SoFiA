#pragma once
#define HEADER 1

extern void uniform_filter_1d_GPU(double *in_cube, double *out_cube, int cube_z, int cube_y, int cube_x, int kz);
