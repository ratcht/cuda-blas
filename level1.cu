#include <iostream>

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


int main() {
  const int n = 10;
  const float a = 1;
  const float x[n] = {1,2,3,4,5,6,7,8,9,10};
  const int incx = 1;
  float y[n] = {2,1,2,1,2,1,2,1,2,1}; 
  const int incy = 1;

  printArray(x, n);
  printArray(y, n);


  saxpy(n,a,x,incx,y,incy);

  printArray(y, n);


  return 0;
}