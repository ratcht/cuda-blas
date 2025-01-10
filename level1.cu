#include <iostream>
#include <stdio.h>

void printArray(const float *array, int size) {
  printf("[");
  for (int i = 0; i < size; i++) {
    printf("%.2f", array[i]);  // Changed %d to %.2f for floating-point numbers
    if (i < size - 1) {
      printf(", ");
    }
  }
  printf("]\n");
}


template <typename T>
__global__ void cuda_axpy(const int n, const T a, const T* x, const int incx, T* y, const int incy) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    y[i*incy] = a*x[i*incx] + y[i*incy];
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

int main() {
  const int n = 10;
  const float a = 1;
  const float x[n] = {1,2,-3,4,5,6,7,8,-9,10};
  const int incx = 1;
  float y[n] = {2,1,2,1,2,1,2,1,2,1}; 
  const int incy = 1;

  //printArray(x, n);
  //printArray(y, n);


  float val = sasum(x,n);
  std::cout<<"val: "<<val<<std::endl;
 // printArray(y, n);


  return 0;
}