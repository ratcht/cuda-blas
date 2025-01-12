#include <stdio.h>
#include <math.h>
#include "blas.cuh"

#define ASSERT(condition, message) \
    do { \
        if (!(condition)) { \
            printf("Test failed: %s\n", message); \
            return 1; \
        } \
    } while (0)

#define ASSERT_FLOAT_EQ(a, b, message) \
    ASSERT(fabs(a - b) < 1e-5f, message)

int test_saxpy() {
    const int n = 5;
    const float alpha = 2.0f;
    float x[5] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    float y[5] = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
    float expected[5] = {3.0f, 5.0f, 7.0f, 9.0f, 11.0f};
    
    saxpy(n, alpha, x, 1, y, 1);
    
    for (int i = 0; i < n; i++) {
        ASSERT_FLOAT_EQ(y[i], expected[i], "SAXPY result mismatch");
    }
    
    return 0;
}

int test_sscal() {
    const int n = 5;
    const float alpha = 3.0f;
    float x[5] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    float expected[5] = {3.0f, 6.0f, 9.0f, 12.0f, 15.0f};
    
    sscal(n, alpha, x, 1);
    
    for (int i = 0; i < n; i++) {
        ASSERT_FLOAT_EQ(x[i], expected[i], "SSCAL result mismatch");
    }
    
    return 0;
}

int test_sasum() {
    const int n = 5;
    float x[5] = {1.0f, -2.0f, 3.0f, -4.0f, 5.0f};
    float expected = 15.0f;  // |1| + |-2| + |3| + |-4| + |5|
    
    float result = sasum(x, n);
    ASSERT_FLOAT_EQ(result, expected, "SASUM result mismatch");
    
    return 0;
}

int main() {
    printf("Running tests...\n");
    
    if (test_saxpy() == 0) {
        printf("SAXPY test passed\n");
    }
    
    if (test_sscal() == 0) {
        printf("SSCAL test passed\n");
    }
    
    if (test_sasum() == 0) {
        printf("SASUM test passed\n");
    }
    
    printf("All tests completed successfully!\n");
    return 0;
}