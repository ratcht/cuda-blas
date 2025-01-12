# CUDA-BLAS

This is my implementation of BLAS (Basic Linear Algebra Subprogram) in CUDA.

## Features

### Level 1 (L1) Operations - In Progress
- **Vector addition** (`axpy`)
- **Scaling** (`scal`)
- **Absolute Sum** (`asum`)
- + more...

### Level 2 (L2) and Level 3 (L3) Operations - Coming Soon
- **Level 2 (Matrix-Vector operations):**
  - `gemv` (General Matrix-Vector Multiplication)
  - `ger` (General Rank-1 Update)
  - + more...
- **Level 3 (Matrix-Matrix operations):**
  - `gemm` (General Matrix-Matrix Multiplication)
  - `trsm` (Triangular Solve with Multiple Right-Hand Sides)
  - + more...
      
## 📂 Repository Structure
```
├── src/                # CUDA source files
├── include/            # Header files
├── examples/           # Usage examples
├── tests/              # Unit tests
└── README.md           # Project overview
```

## Getting Started

### Prerequisites
- **CUDA Toolkit** (Version 11.0 or higher recommended)
- **C++ Compiler** (GCC or MSVC)
- **CMake** (Optional, for build system)

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/ratcht/cuda-blas.git
   cd cuda-blas
   ```
2. Compile the project:
   ```bash
   make
   ```

### Usage
Run the example programs to see the implemented BLAS operations in action:
```bash
./main
```

## Testing
To validate the operations, run the test suite:
```bash
cd tests
./run_tests
```

## Goals
- Complete **L2 and L3 operations**.
- Provide **benchmarking tools** for comparing performance.

---



