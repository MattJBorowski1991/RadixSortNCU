NVCC = nvcc
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

SRCS = $(DRIVERS_DIR)/main.cu \
	   $(KERNELS_DIR)/bitonic.cu \
	   $(KERNELS_DIR)/radix_v1.cu

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(SRCS)
	@mkdir -p $(BIN_DIR)
	$(NVCC) $(NVCC_FLAGS) $(NVCC_GENCODE) $(SRCS) -o $(TARGET)

clean:
	rm -rf $(BIN_DIR)
	rm -f $(KERNELS_DIR)/*.o $(DRIVERS_DIR)/*.o $(UTILS_DIR)/*.o