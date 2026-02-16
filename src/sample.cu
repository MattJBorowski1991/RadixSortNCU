#include "sample.h"
#include <cuda_runtime.h>
#include <cstdint>

constexpr int WarpsInBlock = 32;
constexpr int threads = (32 * WarpsInBlock);

//Sample the token based on prefix sum of final (normalized, decreasing) probabilities and provided seed
__device__ __forceinline__ float lcg_uniform(unsigned int seed) {
    unsigned int state = seed;
    state = (1664525u * state + 1013904223u);  // LCG step
    return (float)state / 4294967296.0f;       // Normalize to [0,1)
}

__global__ void warp_sample(
    const float* __restrict__ prefix_sum_probs,
    const int* seed,
    int *sampled_token,
    int vocab_size
){
    int batch = blockIdx.z;
    int lane = threadIdx.x % 32;

    int seed_val = seed[batch];
    float sampled_prob = lcg_uniform(seed_val);

    const float* batch_prefix = prefix_sum_probs + batch * vocab_size;

    for(int i = 0; i < vocab_size; i += 32){
        int idx = i + lane;
        float val = 1.1f;
        if(idx < vocab_size) val = batch_prefix[idx];
        uint32_t mask = __ballot_sync(0xFFFFFFFFu, val >= sampled_prob);

        if (mask != 0){
            if (lane == (__ffs(mask) - 1)) {
                sampled_token[batch] = idx;
            }
            return;
        }
    }
}



extern "C" void solve_sample(const float* d_p_n_nucleus, const int* seed, int* sampled_token, int max_nucleus_size, const unsigned int* desc_indices, int vocab_size, int num_batches){
    
    dim3 sample_grid(1, 1, num_batches);
    int* nucleus_sampled_token_alloc = nullptr;
    cudaMalloc(&nucleus_sampled_token_alloc, num_batches * sizeof(int));    
    warp_sample<<<sample_grid, threads>>>(d_p_n_nucleus, seed, nucleus_sampled_token_alloc, max_nucleus_size);
    
    cudaDeviceSynchronize();

    // Collect results
    // Map sampled nucleus indices back to original vocabulary token indices
    // (warp_sample returns indices into the nucleus subset, not the full vocabulary)
    for(int batch = 0; batch < num_batches; ++batch) {
      
        // Copy sampled index to host and map back to original token
        int sampled_nucleus_idx = 0;
        cudaMemcpy(&sampled_nucleus_idx, nucleus_sampled_token_alloc + batch, sizeof(int), cudaMemcpyDeviceToHost);
        
        unsigned int original_token = 0;
        const unsigned int* batch_d_indices_in = desc_indices + batch * vocab_size;
        cudaMemcpy(&original_token, batch_d_indices_in + sampled_nucleus_idx, sizeof(unsigned int), cudaMemcpyDeviceToHost);
        
        cudaMemcpy(sampled_token + batch, &original_token, sizeof(unsigned int), cudaMemcpyHostToDevice);
    }

    cudaFree(nucleus_sampled_token_alloc);

}