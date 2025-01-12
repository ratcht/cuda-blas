# CUDA BLAS

A CUDA implementation of Basic Linear Algebra Subprograms (BLAS).

## Current Features

### Level 1 BLAS Operations
- `axpy`: Compute `y = αx + y`
- `scal`: Compute `x = αx`
- `asum`: Compute the sum of absolute values

All operations currently support single-precision floating point (float) with plans to extend to other data types.

## Requirements
- CUDA Toolkit
- C++ Compiler with C++11 support
- Make

## Project Structure
```
cuda-blas/
├── include/          # Header files
├── src/             # Implementation files
├── examples/        # Example usage
└── tests/           # Test files
```

## Building

```bash
# Build everything
make all

# Run examples
./bin/level1_examples

# Clean build files
make clean
```

## Usage Example

```cpp
#include "blas.cuh"

int main() {
  const int n = 5;
  const float alpha = 2.0f;
  float x[] = {1, 2, 3, 4, 5};
  float y[] = {1, 1, 1, 1, 1};

  // Compute y = αx + y
  axpy(n, alpha, x, 1, y, 1);
  
  return 0;
}
```

## Planned Features
- Support for double precision
- Support for complex numbers
- Additional Level 1 BLAS operations
- Implementation of Level 2 and 3 BLAS operations

---

Feel free to reach out with questions or feedback!