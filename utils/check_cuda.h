#ifndef CUDA_UTILS_H
#define CUDA_UTILS_H

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

#endif // CUDA_UTILS_H