NVCC = nvcc

NVCC_FLAGS = -O3 -lineinfo -Xcompiler -Wall

NVCC_ARCH ?= 75

NVCC_GENCODE = -gencode=arch=compute_$(NVCC_ARCH),code=sm_$(NVCC_ARCH) \
		       -gencode=arch=compute_$(NVCC_ARCH),code=compute_$(NVCC_ARCH)

DRIVERS_DIR = drivers
KERNELS_DIR = kernels
UTILS_DIR = utils
BIN_DIR = bin

TARGET = $(BIN_DIR)/profile_harness
TEST_TARGET = $(BIN_DIR)/test_sort

SRCS = $(DRIVERS_DIR)/main.cu \
	   $(KERNELS_DIR)/bitonic.cu \
	   $(KERNELS_DIR)/radix_v1.cu \
	   $(KERNELS_DIR)/radix_v2.cu \
	   $(KERNELS_DIR)/radix_v3.cu

TEST_SRCS = tests/test_sort.cu \
	    $(KERNELS_DIR)/bitonic.cu \
	    $(KERNELS_DIR)/radix_v1.cu \
	    $(KERNELS_DIR)/radix_v2.cu \
		$(KERNELS_DIR)/radix_v3.cu

.PHONY: all clean test_sort

all: $(TARGET) $(TEST_TARGET)

$(TARGET): $(SRCS)
	@mkdir -p $(BIN_DIR)
	$(NVCC) $(NVCC_FLAGS) $(NVCC_GENCODE) $(SRCS) -o $(TARGET)

$(TEST_TARGET): $(TEST_SRCS)
	@mkdir -p $(BIN_DIR)
	$(NVCC) $(NVCC_FLAGS) $(NVCC_GENCODE) $(TEST_SRCS) -o $(TEST_TARGET)

test_sort: $(TEST_TARGET)

clean:
	rm -rf $(BIN_DIR)
	rm -f $(KERNELS_DIR)/*.o $(DRIVERS_DIR)/*.o $(UTILS_DIR)/*.o