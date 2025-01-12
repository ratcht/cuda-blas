# Compiler
NVCC = nvcc

# Files to compile
SRCS = main.cu l1.cu

# Default target
all:
	$(NVCC) $(SRCS)

# Clean up build files
clean:
	rm -f $(TARGET)
