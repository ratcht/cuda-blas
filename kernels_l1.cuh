#ifndef LEVEL1_GLOBAL_H
#define LEVEL1_GLOBAL_H

#include <iostream>
#include <stdio.h>

// Function declarations
inline void printArray(const float* array, int size) {
  printf("[");
  for (int i = 0; i < size; i++) {
    printf("%.2f", array[i]);
    if (i < size - 1) {
      printf(", ");
    }
  }
  printf("]\n");
}

// CUDA kernels
template <typename T>
__global__ void cuda_axpy(const int n, const T a, const T* x, const int incx, T* y, const int incy) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    y[i * incy] = a * x[i * incx] + y[i * incy];
  }
}

template <typename T>
__global__ void cuda_scal(const int n, const T a, T* x, const int incx) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    x[i * incx] *= a;
  }
}

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

#endif  // LEVEL1_GLOBAL_H