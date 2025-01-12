#ifndef CUDA_AXPY_H
#define CUDA_AXPY_H

void saxpy(const int n, const float a, const float* x, const int incx, float* y, const int incy);

#endif // CUDA_AXPY_H