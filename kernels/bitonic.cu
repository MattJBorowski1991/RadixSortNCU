#include <cuda_runtime.h>
#include "../utils/check_cuda.h"
#include <assert.h>
#include <stdio.h>

constexpr int WarpsInBlock = 32;
constexpr int threads = (32 * WarpsInBlock);

template<int THREADS>
__global__ void init_indices(
    unsigned int *indices,
    int vocab_size, 
    int num_batches
){
    const int gid = blockIdx.x * THREADS + threadIdx.x;
    const int batch = blockIdx.z;

    unsigned int *indices_batch = indices + batch * vocab_size;

    if(gid < vocab_size){
        indices_batch[gid] = gid;
    }
}

template<int THREADS>
__global__ void bitonic_sort_step(
    unsigned int *input, unsigned int* input_indices, 
    unsigned int *output, unsigned int *output_indices,
    int j, int k, int vocab_size, 
    int num_batches)
{

    int batch = blockIdx.z;
    unsigned int *input_batch = input + batch * vocab_size;
    unsigned int *input_indices_batch = input_indices + batch * vocab_size;
    unsigned int *output_batch = output + batch * vocab_size;
    unsigned int *output_indices_batch = output_indices + batch * vocab_size;

    int i = blockIdx.x * THREADS + threadIdx.x;
    if(i >= vocab_size) return;
    int ixj = i ^ j;
    if(ixj >= vocab_size || ixj <= i) return;

    bool ascending = ((i & k) == 0);

    unsigned int a = input_batch[i];
    unsigned int b = input_batch[ixj];

    // // For descending:
    // if( (ascending && a < b) || (!ascending && a > b) )
    if( (ascending && a > b) || (!ascending && a < b) ){
        output_batch[i] = b;
        output_batch[ixj] = a;
        output_indices_batch[i] = input_indices_batch[ixj];
        output_indices_batch[ixj] = input_indices_batch[i];
    } else {
        output_batch[i] = a;
        output_batch[ixj] = b;
    }
}

// bitonic sort requires inputs of size 2^N
extern "C" void solve_bitonic(
    unsigned int *input,
    unsigned int *output, unsigned int *output_indices, 
    int vocab_size, int num_batches){
    
    cudaEvent_t start;
    cudaEvent_t stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    const int blocksx = (vocab_size + threads - 1) / threads;
    dim3 grid(blocksx, 1, num_batches);

    size_t total_bytes = vocab_size * num_batches * sizeof(unsigned int);

    unsigned int *d_in, *d_out, *d_indices_in, *d_indices_out;
    cudaMalloc(&d_in, total_bytes);
    cudaMalloc(&d_out, total_bytes);
    cudaMalloc(&d_indices_in, total_bytes);
    cudaMalloc(&d_indices_out, total_bytes);

    cudaMemcpy(d_in, input, total_bytes, cudaMemcpyDeviceToDevice);

    init_indices<threads><<<grid, threads>>>(d_indices_in, vocab_size, num_batches);
    init_indices<threads><<<grid, threads>>>(d_indices_out, vocab_size, num_batches);

    for(int k = 2; k <= vocab_size; k <<= 1){
        for(int j = k >> 1; j > 0; j >>= 1){
            CHECK_CUDA(bitonic_sort_step<threads><<<grid, threads>>>(d_in, d_indices_in, d_out, d_indices_out, j, k, vocab_size, num_batches));
            CHECK_CUDA(cudaDeviceSynchronize());
            std::swap(d_in, d_out);
            std::swap(d_indices_in, d_indices_out);
        }
    }

    cudaMemcpy(output, d_in, total_bytes, cudaMemcpyDeviceToDevice);
    cudaMemcpy(output_indices, d_indices_in, total_bytes, cudaMemcpyDeviceToDevice);

    cudaFree(d_in);
    cudaFree(d_out);
    cudaFree(d_indices_in);
    cudaFree(d_indices_out);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}
