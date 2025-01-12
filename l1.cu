#include "l1.cuh"


void saxpy(const int n, const float a, const float* x, const int incx, float* y, const int incy) {
  int block_size = 256;
  int grid_size = (n + block_size - 1) / block_size;

  float* d_x;
  float* d_y;

  // Allocate device memory
  cudaMalloc((void**)&d_x, n * sizeof(float));
  cudaMalloc((void**)&d_y, n * sizeof(float));

  // Copy data to device
  cudaMemcpy(d_x, x, n * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_y, y, n * sizeof(float), cudaMemcpyHostToDevice);

  cuda_axpy<<<grid_size, block_size>>>(n, a, d_x, incx, d_y, incy);

  cudaDeviceSynchronize();

  // Copy result back to host
  cudaMemcpy(y, d_y, n * sizeof(float), cudaMemcpyDeviceToHost);
}

void sscal(const int n, const float a, float* x, const int incx) {
  int block_size = 256;
  int grid_size = (n + block_size - 1) / block_size;

  float* d_x;

  // Allocate device memory
  cudaMalloc((void**)&d_x, n * sizeof(float));

  // Copy data to device
  cudaMemcpy(d_x, x, n * sizeof(float), cudaMemcpyHostToDevice);

  cuda_scal<<<grid_size, block_size>>>(n, a, d_x, incx);

  cudaDeviceSynchronize();

  // Copy result back to host
  cudaMemcpy(x, d_x, n * sizeof(float), cudaMemcpyDeviceToHost);
}


float sasum(const float* h_x, int n) {
  const int BLOCK_SIZE = 256;  // Number of threads per block
  int num_blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

  // Allocate device memory
  float* d_x;
  float* d_partial_sums;
  cudaMalloc(&d_x, n * sizeof(float));
  cudaMalloc(&d_partial_sums, num_blocks * sizeof(float));

  // Copy data to device
  cudaMemcpy(d_x, h_x, n * sizeof(float), cudaMemcpyHostToDevice);

  cuda_asum<<<num_blocks, BLOCK_SIZE, BLOCK_SIZE * sizeof(float)>>>(d_x, d_partial_sums, n);

  // Copy partial sums back to host
  float* h_partial_sums = new float[num_blocks];
  cudaMemcpy(h_partial_sums, d_partial_sums, num_blocks * sizeof(float), cudaMemcpyDeviceToHost);

  // Final reduction on the host
  float result = 0.0f;
  for (int i = 0; i < num_blocks; ++i) {
    result += h_partial_sums[i];
  }

  delete[] h_partial_sums;
  cudaFree(d_x);
  cudaFree(d_partial_sums);

  return result;
}
