#ifndef LEVEL1_H
#define LEVEL1_H

#include <iostream>
#include <stdio.h>
#include "kernels_l1.cuh"


// Floats
void saxpy(const int n, const float a, const float* x, const int incx, float* y, const int incy);

void sscal(const int n, const float a, float* x, const int incx);

float sasum(const float* h_x, int n);


#endif  // LEVEL1_H