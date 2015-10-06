.PHONY: build

build:
	nvcc -c -m64 -arch=sm_20 -o gaussianSeparable_kernel.o -Xcompiler -fPIC gaussianSeparable_kernel.cu
	gcc gaussianSeparable_kernel.o -std=c99 -shared -fPIC -O3 -o SCfinder_mem.so -lcudart -L/usr/local/cuda/lib64 SCfinder_mem.c
	gcc -shared -O3 -o histogram.so histogram.c -std=c99

test:
	py.test
