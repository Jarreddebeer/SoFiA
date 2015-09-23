#pragma once
#define HEADER 1
extern void gaussian_filter_GPU(double *h_in_cube, double *h_out_cube, size_t cube_z, size_t cube_y, size_t cube_x, size_t ky, size_t kx);
