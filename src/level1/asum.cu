#include "level1/asum.cuh"
#include "utils/cuda_utils.cuh"

template <typename T>
__global__ void cuda_asum(const T* d_x, T* d_partial_sums, int n) {
  extern __shared__ T shared_data[];
  int tid = threadIdx.x;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  // Initialize shared memory
  shared_data[tid] = (idx < n) ? fabsf(d_x[idx]) : 0.0f;
  __syncthreads();

  // Perform reduction in shared memory
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      shared_data[tid] += shared_data[tid + stride];
    }
    __syncthreads();
  }

  // Write the block's partial sum to global memory
  if (tid == 0) {
    d_partial_sums[blockIdx.x] = shared_data[0];
  }
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

  // Clean up
  delete[] h_partial_sums;
  cudaFree(d_x);
  cudaFree(d_partial_sums);

  return result;
}