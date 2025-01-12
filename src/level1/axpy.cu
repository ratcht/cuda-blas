#include "level1/axpy.cuh"
#include "utils/cuda_utils.cuh"

template <typename T>
__global__ void cuda_axpy(const int n, const T a, const T* x, const int incx, T* y, const int incy) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    y[i * incy] = a * x[i * incx] + y[i * incy];
  }
}

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

  // Free device memory
  cudaFree(d_x);
  cudaFree(d_y);
}