#include "level1/scal.cuh"
#include "utils/cuda_utils.cuh"

template <typename T>
__global__ void cuda_scal(const int n, const T a, T* x, const int incx) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    x[i * incx] *= a;
  }
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

  // Free device memory
  cudaFree(d_x);
}