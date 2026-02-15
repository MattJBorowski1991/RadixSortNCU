#include "reverse.h"
#include <cuda_runtime.h>

constexpr int WarpsInBlock = 32;
constexpr int threads = 32 * WarpsInBlock;

template<int THREADS>
__global__ void uint32_to_fp32_kernel(
    const unsigned int* __restrict__ input,
    float* __restrict__ output,
    int N
){
    int gid = blockIdx.x * THREADS + threadIdx.x;
    int batch = blockIdx.z;
    if(gid >= N) return;
    const unsigned int* input_batch = input + batch * N;
    float* output_batch = output + batch * N;
    unsigned int i = input_batch[gid];
    // convert uint32 in [0, 2^32-1] back to float in [0,1]
    float v = __uint2float_rn(i) / 4294967295.0f;
    output_batch[gid] = v;
}

template<int THREADS>
__global__ void reverse_array(
    unsigned int* input,
    unsigned int* input_indices,
    int N
){
    const int gid = blockIdx.x * THREADS + threadIdx.x;
    const int batch = blockIdx.z;
    unsigned int* input_batch = input + batch * N;
    unsigned int* input_indices_batch = input_indices + batch * N;
    int halfN = (N + 1) / 2;
    if(gid < halfN){
        unsigned int left_element = input_batch[gid];
        unsigned int left_idx = input_indices_batch[gid];
        unsigned int right_element = input_batch[N - 1 - gid];
        unsigned int right_idx = input_indices_batch[N - 1 - gid];
        input_batch[gid] = right_element;
        input_indices_batch[gid] = right_idx;
        input_batch[N - 1 - gid] = left_element;
        input_indices_batch[N - 1 - gid] = left_idx;
    }
}


extern "C" void solve_reverse(unsigned int* asc_g_probs, unsigned int* asc_indices, float* desc_g_probs, unsigned int* desc_indices, int vocab_size, int num_batches)
{
  
    int rev_blocks = (vocab_size + threads - 1) / threads;
    dim3 rev_grid(rev_blocks, 1, num_batches);
    reverse_array<threads><<<rev_grid, threads>>>(asc_g_probs, asc_indices, vocab_size);
    
    // float* desc_g_probs = nullptr;
    // cudaMalloc(&desc_g_probs, vocab_size * num_batches * sizeof(float));
    
    int blocksx = (vocab_size + threads - 1) / threads;
    dim3 grid(blocksx, 1, num_batches);
    //convert back to float
    uint32_to_fp32_kernel<threads><<<grid, threads>>>(asc_g_probs, desc_g_probs, vocab_size);
    cudaDeviceSynchronize();

    cudaMemcpy(desc_indices, asc_indices, num_batches * vocab_size * sizeof(unsigned int), cudaMemcpyDeviceToDevice);

}
