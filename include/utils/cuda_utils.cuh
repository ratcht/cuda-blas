#ifndef CUDA_UTILS_H
#define CUDA_UTILS_H

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
__global__ void cuda_axpy(const int n, const T a, const T* x, const int incx, T* y, const int incy);

template <typename T>
__global__ void cuda_scal(const int n, const T a, T* x, const int incx);

template <typename T>
__global__ void cuda_asum(const T* d_x, T* d_partial_sums, int n);

#endif // CUDA_UTILS_H