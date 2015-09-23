//void SCfinder_mem(double *in_cube, size_t cube_z, size_t cube_y, size_t cubex, int *kernels, size_t kern_size);
void SCfinder_mem(double *in_cube, size_t cube_z, size_t cube_y, size_t cubex, int *kernel);
void gaussian_filter(double *in_cube, double *out_cube, size_t cube_z, size_t cube_y, size_t cube_x, size_t ky, size_t kx);
void test_cuda();
void uniform_filter_1d(double *in_cube, double *out_cube,  size_t cube_z, size_t cube_y, size_t cube_x, size_t kz);
