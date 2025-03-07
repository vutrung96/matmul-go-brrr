# Compiler settings
NVCC := /usr/local/cuda-12.4/bin/nvcc
CFLAGS := -std=c++11 -O3
CUDA_PATH := /usr/local/cuda-12.4
INCLUDES := -I$(CUDA_PATH)/include
LDFLAGS := -L$(CUDA_PATH)/lib64 -lcudart -lcublas

# Source and target files
SRC := benchmark.cu
TARGET := matmul_benchmark

# Default target
all: $(TARGET)

# Compile the benchmark
$(TARGET): $(SRC)
	$(NVCC) $(CFLAGS) $(INCLUDES) $(SRC) -o $(TARGET) $(LDFLAGS)

# Run the benchmark
run: $(TARGET)
	./$(TARGET)

# Clean build files
clean:
	rm -f $(TARGET)

.PHONY: all run clean 