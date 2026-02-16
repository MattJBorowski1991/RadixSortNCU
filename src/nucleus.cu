#include "nucleus.h"
#include <cuda_runtime.h>
#include <vector>

constexpr int WarpsInBlock = 32;
constexpr int threads = (32 * WarpsInBlock);

template<int THREADS>
__global__ void prefix_sum(
    const float* __restrict__ input,
    float* __restrict__ output,
    float* __restrict__ blockSums,
    int N
){  
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int batch = blockIdx.z;
    const float* input_batch = input + batch * N;
    float* output_batch = output + batch * N;
    int block_start = bid * THREADS;
    int gid = block_start + tid;

    __shared__ float s[THREADS];
    if(gid < N){
        s[tid] = input_batch[gid];
    }else{
        s[tid] = 0.0f;
    }
    __syncthreads();

    for(int offset = 1; offset < THREADS; offset <<= 1){
        float add = 0.0f;
        if(tid >= offset) add = s[tid - offset];
        __syncthreads();
        s[tid] += add;
        __syncthreads();
    }

    if(gid < N) output_batch[gid] = s[tid];
    __syncthreads();

    if(tid == 0) {
        int out_idx = batch * ((N + THREADS - 1) / THREADS) + bid;
        blockSums[out_idx] = s[THREADS - 1];
    }
}

template<int THREADS>
__global__ void add_block_offset(
    const float* __restrict__ input,
    const float* __restrict__ offset,
    float* __restrict__ output,
    int N
){  
    int bid = blockIdx.x;
    int batch = blockIdx.z;
    int gid = bid * THREADS + threadIdx.x;
    if(gid >= N) return;
    const float* input_batch = input + batch * N;
    const float* offset_batch = offset + batch * ((N + THREADS - 1) / THREADS);
    float* output_batch = output + batch * N;
    float off = (bid == 0) ? 0.0f : offset_batch[bid - 1];
    output_batch[gid] = input_batch[gid] + off;
}


extern "C" void solve_nucleus_part1(const float* desc_g_probs, int* h_nucleus_sizes, float* h_global_probs_sums, int* max_nucleus_size, int vocab_size, int num_batches, const float* p){
    
    // We loop over batches here because each batch can have different (small) nucleus size
    // Allocate per-batch tracking arrays
    // std::vector<float> h_global_probs_sums(num_batches);
    // std::vector<unsigned int*> d_nucleus_arrays(num_batches);
    // std::vector<float*> d_n_nucleus_arrays(num_batches);
    // std::vector<float*> d_p_n_nucleus_arrays(num_batches);
    
    *max_nucleus_size = 0;
    int blocksx = (vocab_size + threads - 1) / threads;
    dim3 grid(blocksx, 1, num_batches);

    for(int batch = 0; batch < num_batches; ++batch) {
        // STEP 3.1: Find nucleus size via prefix sum on sorted probabilities
        float* d_blockSums = nullptr;
        cudaMalloc(&d_blockSums, blocksx * sizeof(float));
        cudaMemset(d_blockSums, 0, blocksx * sizeof(float));
        
        float* d_per_block_sums = nullptr;
        cudaMalloc(&d_per_block_sums, sizeof(float) * vocab_size);
        
        // Prefix sum on batch's sorted probabilities
        const float* batch_desc_g_probs = desc_g_probs + batch * vocab_size;
        prefix_sum<threads><<<blocksx, threads>>>(batch_desc_g_probs, d_per_block_sums, d_blockSums, vocab_size);
        
        // Copy block sums and find first block where cumulative sum >= p
        std::vector<float> h_blockSums(blocksx);
        cudaMemcpy(h_blockSums.data(), d_blockSums, sizeof(float)*blocksx, cudaMemcpyDeviceToHost);
        
        float p_val = 0.0f;
        cudaMemcpy(&p_val, p + batch, sizeof(float), cudaMemcpyDeviceToHost);
        
        int target_block = -1;
        int target_index = -1;
        float running = 0.0f;
        for (int b = 0; b < blocksx; ++b) {
            running += h_blockSums[b];
            if (running >= p_val) { target_block = b; break; }
        }
        
        if (target_block == -1) {
            target_block = blocksx - 1;
            target_index = vocab_size - 1;
        }
        
        float prev_block_prefix = running - h_blockSums[target_block];
        
        int blockStart = target_block * threads;
        int elements_in_block = min(vocab_size - blockStart, threads);
        std::vector<float> h_targetBlockPrefix(elements_in_block);
        cudaMemcpy(h_targetBlockPrefix.data(), d_per_block_sums + blockStart, sizeof(float) * elements_in_block, cudaMemcpyDeviceToHost);
        
        float need = p_val - prev_block_prefix;
        auto it = std::lower_bound(h_targetBlockPrefix.begin(), h_targetBlockPrefix.end(), need);
        int local_idx = (it == h_targetBlockPrefix.end()) ? (elements_in_block - 1) : int(it - h_targetBlockPrefix.begin());
        target_index = blockStart + local_idx;
        
        int nucleus_size = target_index + 1;
        nucleus_size = std::min(nucleus_size, 1000);  // Cap nucleus to top-1000
        
        h_nucleus_sizes[batch] = nucleus_size;
        *max_nucleus_size = max(*max_nucleus_size, nucleus_size);
        
        // Recompute sum based on capped nucleus_size by reading the actual probabilities
        float global_probs_sum = 0.0f;
        std::vector<float> h_sorted_probs(nucleus_size);
        cudaMemcpy(h_sorted_probs.data(), desc_g_probs + batch * vocab_size, nucleus_size * sizeof(float), cudaMemcpyDeviceToHost);
        for(int i = 0; i < nucleus_size; i++) {
            global_probs_sum += h_sorted_probs[i];
        }
        h_global_probs_sums[batch] = global_probs_sum;
        
        cudaFree(d_blockSums);
        cudaFree(d_per_block_sums);
    }
    cudaDeviceSynchronize(); 
}

extern "C" void solve_nucleus_part2(const float* desc_g_probs, int* h_nucleus_sizes, float* d_nucleus, int max_nucleus_size, int vocab_size, int num_batches)
{
    
    // Copy nucleus portions from sorted probs to nucleus allocation for all batches
    for(int batch = 0; batch < num_batches; ++batch) {
        int nucleus_size = h_nucleus_sizes[batch];
        int nucleus_bytes = nucleus_size * sizeof(float);
        
        // Copy from sorted probs device buffer to nucleus allocation (device-to-device)
        cudaMemcpy(d_nucleus + batch * max_nucleus_size, 
                   desc_g_probs + batch * vocab_size, 
                   nucleus_bytes, cudaMemcpyDeviceToDevice);
    }

}