#include <stdio.h>
#include "blas.cuh"

int main() {
    const int n = 10;
    const float alpha = 3.0f;
    
    // Initialize arrays
    float x[10] = {1,2,-3,4,5,6,7,8,-9,10};
    float y[10] = {2,1,2,1,2,1,2,1,2,1};

    // Test SSCAL
    printf("Original array y:\n");
    printArray(y, n);
    
    sscal(n, alpha, y, 1);
    
    printf("\nAfter SSCAL (y = alpha * y):\n");
    printArray(y, n);

    // Test SAXPY
    printf("\nPerforming SAXPY (y = alpha * x + y):\n");
    saxpy(n, alpha, x, 1, y, 1);
    printArray(y, n);

    // Test SASUM
    float sum = sasum(x, n);
    printf("\nAbsolute sum of x: %.2f\n", sum);

    return 0;
}