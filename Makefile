# Compiler and flags
NVCC = nvcc
NVCC_FLAGS = -O3 -I./include

# Directories
SRC_DIR = src
INCLUDE_DIR = include
BUILD_DIR = build
BIN_DIR = bin
OBJ_DIR = $(BUILD_DIR)/obj
LIB_DIR = $(BUILD_DIR)/lib

# Source files
LEVEL1_SRCS = $(wildcard $(SRC_DIR)/level1/*.cu)
UTILS_SRCS = $(wildcard $(SRC_DIR)/utils/*.cu)
ALL_SRCS = $(LEVEL1_SRCS) $(UTILS_SRCS)

# Object files
OBJS = $(ALL_SRCS:$(SRC_DIR)/%.cu=$(OBJ_DIR)/%.o)

# Library name
LIB_NAME = libcublas.a
LIB = $(LIB_DIR)/$(LIB_NAME)

# Examples and tests
EXAMPLES_SRC = examples/level1_examples.cu
TESTS_SRC = tests/level1_tests.cu
EXAMPLES_BIN = $(BIN_DIR)/level1_examples
TESTS_BIN = $(BIN_DIR)/level1_tests

# Create directories
$(shell mkdir -p $(OBJ_DIR)/level1 $(OBJ_DIR)/utils $(BIN_DIR) $(LIB_DIR))

# Default target
all: $(LIB) examples tests

# Library
$(LIB): $(OBJS)
	ar rcs $@ $^

# Compile source files
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cu
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

# Examples
examples: $(EXAMPLES_BIN)

$(EXAMPLES_BIN): $(EXAMPLES_SRC) $(LIB)
	$(NVCC) $(NVCC_FLAGS) $< -L$(LIB_DIR) -lcublas -o $@

# Tests
tests: $(TESTS_BIN)

$(TESTS_BIN): $(TESTS_SRC) $(LIB)
	$(NVCC) $(NVCC_FLAGS) $< -L$(LIB_DIR) -lcublas -o $@

# Clean
clean:
	rm -rf $(BUILD_DIR) $(BIN_DIR)

.PHONY: all examples tests clean