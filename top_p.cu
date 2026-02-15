#include <cuda_runtime.h>
#include <vector>
#include <algorithm>
#include <stdio.h>
#include <cfloat>
#include <assert.h>
#include <iostream>
#include <stdint.h>
#include "../prof/cuda_timer.h"
#include "../include/softmax.h"
#include "../include/sort.h"
#include "../include/reverse.h"

#define WarpsInBlock 32
#define threads (32 * WarpsInBlock)


// // // **** PREFIX SUM (FIND TOP-P) **** // // //

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



extern "C" void solve(
    const float* logits,
    const float *p,
    const int* seed, 
    int* sampled_token,
    int vocab_size,
    int num_batches
){  
    
    // CUDA Events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    struct TimingResult {
        const char* name;
        float time_ms;
        long long bytes_read;
        long long bytes_written;
        long long flops;
        
        TimingResult(const char* n, float t, long long br, long long bw, long long f) 
            : name(n), time_ms(t), bytes_read(br), bytes_written(bw), flops(f) {}
        
        void print(float total_time, int vocab_size) const {
            float mb_read = bytes_read / (1024.0f * 1024);
            float mb_write = bytes_written / (1024.0f * 1024);
            float gbs_read = (bytes_read / (1024.0f * 1024 * 1024)) / (time_ms / 1000.0f);
            float gbs_write = (bytes_written / (1024.0f * 1024 * 1024)) / (time_ms / 1000.0f);
            float gflops = (flops / 1e9f) / (time_ms / 1000.0f);
            float total_bytes = bytes_read + bytes_written;
            float arith_intensity = (total_bytes > 0) ? ((flops / 1e9f) / (total_bytes / (1024.0f * 1024 * 1024))) : 0.0f;
            float percent = (total_time > 0) ? (time_ms / total_time * 100.0f) : 0.0f;
            float ns_per_tok = (vocab_size > 0) ? (time_ms * 1000000.0f / vocab_size) : 0.0f;
            
            printf("%-20s | %9.3f | %6.2f%% | %9.2f | %9.3f | %11.1f | %10.3f | %12.1f | %6.1f | %7.3f\n",
                   name, time_ms, percent, ns_per_tok, mb_read, gbs_read, mb_write, gbs_write, gflops, arith_intensity);
        }
    };
    
    std::vector<TimingResult> timings;

    // // // // // STEPS:
    // // // STEP 0:    H2D
    // // // STEP 1:    softmax:    logits -> g_probs
    // // // STEP 2.1.  sort:       g_probs -> asc_g_probs, asc_indices
    // // // STEP 2.2.  reverse:    asc_g_probs, asc_indices -> desc_g_probs, desc_indices
    // // // STEP 3.1.  nucleus:    desc_g_probs -> d_nucleus [via prefix]
    // // // STEP 3.2.  norm:       d_nucleus -> d_n_nucleus
    // // // STEP 3.3.  prefix:     d_n_nucleus -> d_p_n_nucleus
    // // // STEP 4:    sample:     d_p_n_nucleus -> sampled_token

    // assert(vocab_size % threads == 0);

    // Measure H2D transfer time
    cudaEvent_t h2d_start, h2d_stop;
    cudaEventCreate(&h2d_start);
    cudaEventCreate(&h2d_stop);
    
    float h2d_logits_time = 0.0f, h2d_p_time = 0.0f;
    float* d_logits_temp = nullptr;
    float* d_p_temp = nullptr;
    int* d_seed_temp = nullptr;
    
    cudaMalloc(&d_logits_temp, vocab_size * sizeof(float));
    cudaMalloc(&d_p_temp, sizeof(float));
    cudaMalloc(&d_seed_temp, sizeof(int));
    
    cudaEventRecord(h2d_start);
    cudaMemcpy(d_logits_temp, logits, vocab_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaEventRecord(h2d_stop);
    cudaEventSynchronize(h2d_stop);
    cudaEventElapsedTime(&h2d_logits_time, h2d_start, h2d_stop);
    
    cudaEventRecord(h2d_start);
    cudaMemcpy(d_p_temp, p, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_seed_temp, seed, sizeof(int), cudaMemcpyHostToDevice);
    cudaEventRecord(h2d_stop);
    cudaEventSynchronize(h2d_stop);
    cudaEventElapsedTime(&h2d_p_time, h2d_start, h2d_stop);
    
    // Note: using original pointers passed to solve()
    float h2d_total_time = h2d_logits_time + h2d_p_time;
    long long h2d_total_bytes = vocab_size * sizeof(float) + 2 * sizeof(int);


    // // // STEP 1: softmax(logits) = g_probs

    CudaTimer softmax_timer;
    softmax_timer.startEvent();

    float* g_probs = nullptr;
    cudaMalloc(&g_probs, vocab_size * num_batches * sizeof(float));
    
    // Call refactored softmax function
    solve_softmax(logits, g_probs, vocab_size, num_batches);
    
    softmax_timer.stopEvent();
    softmax_timer.sync();
    float time_step1 = softmax_timer.elapsedMs();
    long long bytes_step1 = vocab_size * sizeof(float) * 2; // read logits, write g_probs
    long long flops_step1 = vocab_size * 10; // approximate: exp, div, reduce ops
    timings.push_back({"STEP 1: Softmax", time_step1, bytes_step1, (long long)((long long)vocab_size * sizeof(float)), flops_step1});



    // // // STEP 2.1.: sort: g_probs -> asc_g_probs

    CudaTimer sort_timer;
    sort_timer.startEvent();
    
    unsigned int *asc_g_probs;
    unsigned int *asc_indices;
    cudaMalloc(&asc_g_probs, vocab_size * num_batches * sizeof(unsigned int));  //still before conversion back to fp32
    cudaMalloc(&asc_indices, vocab_size * num_batches * sizeof(unsigned int));

    solve_sort(g_probs, asc_g_probs, asc_indices, vocab_size, num_batches);

    cudaFree(g_probs);

    sort_timer.stopEvent();
    sort_timer.sync();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float time_sort = sort_timer.elapsedMs();
    long long bytes_sort = vocab_size * sizeof(unsigned int) * 2; // read d_in, write d_out
    long long flops_sort = vocab_size * 32; // 32-bit sort
    timings.push_back({"STEP 2.1: Sort", time_sort, bytes_sort, (long long)(vocab_size * sizeof(unsigned int)), flops_sort});
    

    // // // STEP 2.2.: reverse: asc_g_probs -> desc_g_probs

    // Reverse to get descending order - all batches at once
    cudaEventRecord(start);
    
    float* desc_g_probs = nullptr;
    unsigned int* desc_indices = nullptr;
    cudaMalloc(&desc_g_probs, vocab_size * num_batches * sizeof(float));
    cudaMalloc(&desc_indices, vocab_size * num_batches * sizeof(unsigned int));

    solve_reverse(asc_g_probs, asc_indices, desc_g_probs, desc_indices, vocab_size, num_batches);

    cudaFree(asc_g_probs);
    cudaFree(asc_indices);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float time_reverse = 0.0f;
    cudaEventElapsedTime(&time_reverse, start, stop);
    long long bytes_reverse = vocab_size * sizeof(unsigned int) * 2 + vocab_size * sizeof(float);
    long long flops_reverse = vocab_size;
    timings.push_back({"STEP 2.2: Reverse", time_reverse, bytes_reverse, (long long)(vocab_size * sizeof(float)), flops_reverse});


    // // // STEP 3: Nucleus

    // NOTE: We loop over batches here (exception to rule 4) because each batch can have different nucleus size
    // Allocate per-batch tracking arrays
    std::vector<int> h_nucleus_sizes(num_batches);
    std::vector<float> h_global_probs_sums(num_batches);
    std::vector<unsigned int*> d_nucleus_arrays(num_batches);
    std::vector<float*> d_n_nucleus_arrays(num_batches);
    std::vector<float*> d_p_n_nucleus_arrays(num_batches);
    
    int max_nucleus_size = 0;
    int blocksx = (vocab_size + threads - 1) / threads;
    dim3 grid(blocksx, 1, num_batches);
    
    cudaEventRecord(start);
    
    // STEP 3.1: Find nucleus size via prefix sum on sorted probabilities

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
        max_nucleus_size = max(max_nucleus_size, nucleus_size);
        
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
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float time_step31 = 0.0f;
    cudaEventElapsedTime(&time_step31, start, stop);
    long long bytes_step31 = vocab_size * sizeof(float) * num_batches;
    long long flops_step31 = vocab_size * 2 * num_batches;
    timings.push_back({"STEP 3.1. TOP-P*", time_step31, bytes_step31, (long long)(max_nucleus_size * sizeof(float)), flops_step31});
    
    // Now allocate and process with max nucleus size
    constexpr int final_threads = 1024;
    
    cudaEventRecord(start);
    
    // Allocate batched arrays (all sized to max_nucleus_size)
    int max_nucleus_bytes = max_nucleus_size * sizeof(float);
    float* d_nucleus = nullptr;
    float* d_n_nucleus = nullptr;
    float* d_p_n_nucleus = nullptr;
    int* nucleus_sampled_token_alloc = nullptr;
    
    cudaMalloc(&d_nucleus, max_nucleus_bytes * num_batches);
    cudaMalloc(&d_n_nucleus, max_nucleus_bytes * num_batches);
    cudaMalloc(&d_p_n_nucleus, max_nucleus_bytes * num_batches);
    cudaMalloc(&nucleus_sampled_token_alloc, num_batches * sizeof(int));
    
    // Zero-initialize nucleus allocation to handle padding for batches with nucleus_size < max_nucleus_size
    cudaMemset(d_nucleus, 0, max_nucleus_bytes * num_batches);
    
    // Copy nucleus portions from sorted probs for all batches
    cudaDeviceSynchronize();  // Ensure conversion is complete
    
    // Copy nucleus portions from sorted probs to nucleus allocation for all batches
    for(int batch = 0; batch < num_batches; ++batch) {
        int nucleus_size = h_nucleus_sizes[batch];
        int nucleus_bytes = nucleus_size * sizeof(float);
        
        // Copy from sorted probs device buffer to nucleus allocation (device-to-device)
        cudaMemcpy(d_nucleus + batch * max_nucleus_size, 
                   desc_g_probs + batch * vocab_size, 
                   nucleus_bytes, cudaMemcpyDeviceToDevice);
    }
    
    // STEP 3.2: Normalize
    int norm_blocks = (max_nucleus_size + final_threads - 1) / final_threads;
    
    // Need to create per-batch sums array for normalize kernel
    float* d_norm_sums = nullptr;
    cudaMalloc(&d_norm_sums, num_batches * sizeof(float));
    cudaMemcpy(d_norm_sums, h_global_probs_sums.data(), num_batches * sizeof(float), cudaMemcpyHostToDevice);
    
    CudaTimer norm_timer;
    norm_timer.startEvent();
    solve_normalize(d_nucleus, d_n_nucleus, max_nucleus_size, d_norm_sums, vocab_size, num_batches);
    cudaDeviceSynchronize();
    norm_timer.stopEvent();
    norm_timer.sync();
    float time_norm = norm_timer.elapsedMs();
    long long bytes_norm = max_nucleus_size * sizeof(float) * 2 * num_batches; // read nucleus, write normalized
    long long flops_norm = max_nucleus_size * num_batches; // division per element
    timings.push_back({"STEP 3.2: Norm", time_norm, bytes_norm, (long long)(max_nucleus_size * sizeof(float) * num_batches), flops_norm});
    
    // STEP 3.3: Prefix sum for all batches
    float* d_blockSums_alloc = nullptr;
    cudaMalloc(&d_blockSums_alloc, norm_blocks * num_batches * sizeof(float));
    
    dim3 norm_grid(norm_blocks, 1, num_batches);
    CudaTimer samp_timer;
    samp_timer.startEvent();
    prefix_sum<final_threads><<<norm_grid, final_threads>>>(d_n_nucleus, d_p_n_nucleus, d_blockSums_alloc, max_nucleus_size);
    cudaDeviceSynchronize();
    
    // STEP 3.4: Sample for all batches
    dim3 sample_grid(1, 1, num_batches);
    warp_sample<<<sample_grid, final_threads>>>(d_p_n_nucleus, seed, nucleus_sampled_token_alloc, max_nucleus_size);
    cudaDeviceSynchronize();
    samp_timer.stopEvent();
    samp_timer.sync();
    float time_samp = samp_timer.elapsedMs();
    long long bytes_samp = max_nucleus_size * sizeof(float) * num_batches; // read prefix sums
    long long flops_samp = max_nucleus_size * num_batches; // sampling operations
    timings.push_back({"STEP 3.3-4: Samp", time_samp, bytes_samp, (long long)(num_batches * sizeof(int)), flops_samp});
    
    // Collect results
    // Map sampled nucleus indices back to original vocabulary token indices
    // (warp_sample returns indices into the nucleus subset, not the full vocabulary)
    for(int batch = 0; batch < num_batches; ++batch) {
        int nucleus_size = h_nucleus_sizes[batch];
        
        // Copy sampled index to host and map back to original token
        int sampled_nucleus_idx = 0;
        cudaMemcpy(&sampled_nucleus_idx, nucleus_sampled_token_alloc + batch, sizeof(int), cudaMemcpyDeviceToHost);
        
        unsigned int original_token = 0;
        const unsigned int* batch_d_indices_in = asc_indices + batch * vocab_size;
        cudaMemcpy(&original_token, batch_d_indices_in + sampled_nucleus_idx, sizeof(unsigned int), cudaMemcpyDeviceToHost);
        
        cudaMemcpy(sampled_token + batch, &original_token, sizeof(unsigned int), cudaMemcpyHostToDevice);
    }
    
    // Cleanup
    cudaFree(d_nucleus);
    cudaFree(d_n_nucleus);
    cudaFree(d_p_n_nucleus);
    cudaFree(d_blockSums_alloc);
    cudaFree(nucleus_sampled_token_alloc);
    cudaFree(d_norm_sums);

    // Cleanup
    cudaFree(desc_g_probs);
    cudaFree(d_logits_temp);
    cudaFree(d_p_temp);
    cudaFree(d_seed_temp);
    
    // Print timing results
    printf("\n========== TIMING RESULTS ==========\n");
    printf("Name                 | Time (ms) |    %% | ns/tok | Read (MB) | Read (GB/s) | Write (MB) | Write (GB/s) | GFLOPS | AI\n");
    printf("---------------------+-----------+------+--------+----------+-------------+------------+--------------+--------+-------\n");
    
    // Calculate total time including H2D
    float total_time = h2d_total_time;
    for(const auto& result : timings) total_time += result.time_ms;
    
    // Print H2D transfer
    float h2d_percent = (total_time > 0) ? (h2d_total_time / total_time * 100.0f) : 0.0f;
    float h2d_mb = h2d_total_bytes / (1024.0f * 1024);
    float h2d_bw = (h2d_total_bytes / (1024.0f * 1024 * 1024)) / (h2d_total_time / 1000.0f);
    float h2d_ns_per_tok = (vocab_size > 0) ? (h2d_total_time * 1000000.0f / vocab_size) : 0.0f;
    printf("%-20s | %9.3f | %6.2f%% | %9.2f | %9.3f | %11.1f | %10.3f | %12.1f | %6s | %7s\n",
           "H2D Transfer", h2d_total_time, h2d_percent, h2d_ns_per_tok, h2d_mb, h2d_bw, 0.0f, 0.0f, "N/A", "N/A");
    
    // Print each timing with percentage
    for(const auto& result : timings) {
        result.print(total_time, vocab_size);
    }
    
    printf("%-20s-+-----------+------+--------+----------+-------------+------------+--------------+--------+-------\n", "");
    float total_ns_per_tok = (vocab_size > 0) ? (total_time * 1000000.0f / vocab_size) : 0.0f;
    printf("%-20s | %9.3f | 100.0%% | %9.2f |\n", "TOTAL", total_time, total_ns_per_tok);
    printf("==================================================\n\n");

}