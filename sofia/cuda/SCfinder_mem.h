//void SCfinder_mem(double *in_cube, size_t cube_z, size_t cube_y, size_t cubex, int *kernels, size_t kern_size);
void SCfinder_mem(float *in_cube, size_t cube_z, size_t cube_y, size_t cube_x, int *kernel);
void gaussian_filter(float *in_cube, float *out_cube, size_t cube_z, size_t cube_y, size_t cube_x, size_t ky, size_t kx);
void test_cuda(float *h_in_cube, float *h_out_cube, size_t cube_z, size_t cube_y, size_t cube_x, size_t ky, size_t kx);
void test_cuda_uniform(float *h_in_cube, float *h_out_cube, size_t cube_z, size_t cube_y, size_t cube_x, size_t kz);
void uniform_filter_1d(float *in_cube, float *out_cube,  size_t cube_z, size_t cube_y, size_t cube_x, size_t kz);
