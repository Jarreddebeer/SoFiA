#pragma once
#define HEADER 1

extern void gaussian_filter_GPU(double *in_cube, double *out_cube, int cube_z, int cube_y, int cube_x, int ky, int kx);
double* init_out_cube(size_t cube_z, size_t cube_y, size_t cube_x);
