//void SCfinder_mem(float *in_cube, size_t cube_z, size_t cube_y, size_t cubex, int *kernels, size_t kern_size);
void SCfinder_mem(float *in_cube, size_t cube_z, size_t cube_y, size_t cubex, int *kernel);
void gaussian_filter(float *in_cube, float *out_cube, size_t cube_z, size_t cube_y, size_t cube_x, size_t kz, size_t ky, size_t kx);
void uniform_filter_1d(float *in_cube, float *out_cube,  size_t cube_z, size_t cube_y, size_t cube_x, size_t kz);
