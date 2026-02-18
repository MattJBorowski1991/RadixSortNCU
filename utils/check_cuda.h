#ifndef CHECK_CUDA_H
#define CHECK_CUDA_H

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#define CHECK_CUDA(...) do { \
    __VA_ARGS__; \
    cudaError_t err = cudaGetLastError(); \
    if( err != cudaSuccess){ \
        printf("CUDA error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

#endif // CHECK_CUDA