#include <iostream>
#include <math.h>
#include <stdio.h>
#include <time.h>

__global__ void op_single(int n, double r1) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int step = blockDim.x * gridDim.x;
	double total = 0.0;
	for (int i = index; i < n; i += step) {
		total += atan(r1);
	}
	printf("tot single: %lf\n", total);
}

__global__ void op_multi(int n, double r1, float *device_arr) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int step = blockDim.x * gridDim.x;

	for (int i = index; i < n; i += step) {
		device_arr[i] += atan(r1);
	}
}

int main(int argc, char *argv[]) {
	srand(42);
	double r1 = ((double) rand()) / ((double) (RAND_MAX));
	int n_iterations = 4194304;

	// multi thread
		printf("running multi\n");
		int block_size = 1024;

		int num_blocks = n_iterations / block_size;
		int n_threads = num_blocks * block_size;
		int blocks_per_grid = n_iterations / block_size;

		clock_t t = clock();
		size_t mem_size = n_iterations * sizeof(float);
		printf("using byte: %d\n", (int) mem_size);
		float *host_arr = (float *) malloc(mem_size);
		for (int i = 0; i < n_iterations; i++) {
			host_arr[i] = 0.0;
		}

		float *device_arr = NULL;
		cudaMalloc((void **) &device_arr, mem_size);
		cudaMemcpy(device_arr, host_arr, mem_size, cudaMemcpyHostToDevice);

		clock_t t2 = clock();
		printf("num grids: %d, num threads: %d\n", blocks_per_grid, block_size);
		op_multi<<<blocks_per_grid, block_size>>>(n_iterations, r1, device_arr);
		t2 = clock() - t2;

		cudaMemcpy(host_arr, device_arr, mem_size, cudaMemcpyDeviceToHost);
		float sum = 0.0;
		for (int i = 0; i < n_iterations; i++) {
			sum += host_arr[i];
		}
		t = clock() - t;
		printf("tot multi: %f\n", sum);

		cudaFree(device_arr);
		free(host_arr);
		printf("It took GPU multi with malloc: %f s.\n", (((float) t) / 1000000));
		printf("It took GPU multi kernel only: %f s.\n", (((float) t2) / 1000000));

		//single thread
		printf("running single\n");
		 t = clock();
		op_single<<<1, 1>>>(n_iterations, r1);
		cudaDeviceSynchronize();
		printf("It took GPU single %f s.\n", (((float) clock() - t) / 1000000));
	return 0;
}
