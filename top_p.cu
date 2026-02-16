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
#include "../include/nucleus.h"
#include "../include/sample.h"

#define TIMER(timer_name, read_bytes_expr, write_bytes_expr, flops_expr, name_str, code_block) \
do { \
    CudaTimer timer_name; \
    timer_name.startEvent(); \
    code_block; \
    timer_name.stopEvent(); \
    timer_name.sync(); \
    float time = timer_name.elapsedMs(); \
    long long read_bytes = read_bytes_expr; \
    long long write_bytes = write_bytes_expr; \
    long long flops = flops_expr; \
    timings.push_back({name_str, time, read_bytes, write_bytes, flops}); \
} while(0)


template<int THREADS>
__global__ void prefix_sum(const float* __restrict__ input, float* __restrict__ output, float* __restrict__ blockSums, int N);

#define WarpsInBlock 32
#define threads (32 * WarpsInBlock)


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


    // // // STEP 0:    H2D

    CudaTimer h2d_timer;
    h2d_timer.startEvent();

    float h2d_logits_time = 0.0f, h2d_p_time = 0.0f;
    float* d_logits_temp = nullptr;
    float* d_p_temp = nullptr;
    int* d_seed_temp = nullptr;
    
    cudaMalloc(&d_logits_temp, vocab_size * sizeof(float));
    cudaMalloc(&d_p_temp, sizeof(float));
    cudaMalloc(&d_seed_temp, sizeof(int));
    
    cudaMemcpy(d_logits_temp, logits, vocab_size * sizeof(float), cudaMemcpyHostToDevice);   
    cudaMemcpy(d_p_temp, p, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_seed_temp, seed, sizeof(int), cudaMemcpyHostToDevice);
    
    h2d_timer.stopEvent();
    h2d_timer.sync();
    float h2d_time = h2d_timer.elapsedMs();
    
    // Note: using original pointers passed to solve()
    float h2d_total_time = h2d_logits_time + h2d_p_time;
    long long h2d_total_bytes = vocab_size * sizeof(float) + 2 * sizeof(int);

    cudaFree(d_logits_temp);
    cudaFree(d_p_temp);
    cudaFree(d_seed_temp);


    // // // STEP 1: softmax(logits) = g_probs

    float* g_probs = nullptr;
    TIMER(softmax_timer, vocab_size * sizeof(float) * 2LL, (long long)vocab_size * sizeof(float), vocab_size * 10LL, "STEP 1: softmax", {
        cudaMalloc(&g_probs, vocab_size * num_batches * sizeof(float));
        solve_softmax(logits, g_probs, vocab_size, num_batches);
    }); 

    // // // STEP 2.1.: sort: g_probs -> asc_g_probs

    unsigned int *asc_g_probs;
    unsigned int *asc_indices;

    TIMER(sort_timer, vocab_size * sizeof(unsigned int), vocab_size * sizeof(unsigned int), vocab_size * 32 , "STEP 2.1: sort", {
        cudaMalloc(&asc_g_probs, vocab_size * num_batches * sizeof(unsigned int));  //still before conversion back to fp32
        cudaMalloc(&asc_indices, vocab_size * num_batches * sizeof(unsigned int));
        solve_sort(g_probs, asc_g_probs, asc_indices, vocab_size, num_batches);
        cudaFree(g_probs);
    });
    
    // // // STEP 2.2.: reverse: asc_g_probs -> desc_g_probs
    
    float* desc_g_probs = nullptr;
    unsigned int* desc_indices = nullptr;
    TIMER(reverse_timer, vocab_size * sizeof(unsigned int), vocab_size * sizeof(unsigned int), vocab_size, "STEP 2.2: reverse", {
        cudaMalloc(&desc_g_probs, vocab_size * num_batches * sizeof(float));            
        cudaMalloc(&desc_indices, vocab_size * num_batches * sizeof(unsigned int));     
        solve_reverse(asc_g_probs, asc_indices, desc_g_probs, desc_indices, vocab_size, num_batches);
        cudaFree(asc_g_probs);
        cudaFree(asc_indices);
    });

    // // // STEP 3.1: nucleus:    desc_g_probs -> d_nucleus [via prefix] 

    std::vector<int> h_nucleus_sizes(num_batches);
    std::vector<float> h_global_probs_sums(num_batches);
    int max_nucleus_size = 0;
    float* d_nucleus = nullptr;
    size_t max_nucleus_bytes;

    TIMER(nucleus_timer, vocab_size * sizeof(float) * num_batches, (long long)(max_nucleus_size * sizeof(float)), vocab_size * 2LL, "STEP 3.1: nucleus", {
        solve_nucleus_part1(desc_g_probs, h_nucleus_sizes.data(), h_global_probs_sums.data(), &max_nucleus_size, vocab_size, num_batches, p);
        max_nucleus_bytes = max_nucleus_size * sizeof(float);
        cudaMalloc(&d_nucleus, max_nucleus_bytes * num_batches);
        cudaMemset(d_nucleus, 0, max_nucleus_bytes * num_batches);
        solve_nucleus_part2(desc_g_probs, h_nucleus_sizes.data(), d_nucleus, max_nucleus_size, vocab_size, num_batches);
        cudaDeviceSynchronize();
        cudaFree(desc_g_probs);    
    });

    // // // STEP 3.2.  normalize: d_nucleus -> d_n_nucleus
    // // // and
    // // // STEP 3.3.  prefix: d_n_nucleus -> d_p_n_nucleus

    float* d_p_n_nucleus = nullptr;

    TIMER(norm_timer, max_nucleus_size * sizeof(float) * 2LL * num_batches, max_nucleus_size * sizeof(float) * num_batches, max_nucleus_size * num_batches, "STEP 3.2: norm+pref", {
        // step 3.2.
        int norm_blocks = (max_nucleus_size + threads - 1) / threads;
        // Need to create per-batch sums array for normalize kernel
        float* d_norm_sums = nullptr;
        cudaMalloc(&d_norm_sums, num_batches * sizeof(float));
        cudaMemcpy(d_norm_sums, h_global_probs_sums.data(), num_batches * sizeof(float), cudaMemcpyHostToDevice);
        float* d_n_nucleus = nullptr;
        cudaMalloc(&d_n_nucleus, max_nucleus_bytes * num_batches);
        solve_normalize(d_nucleus, d_n_nucleus, max_nucleus_size, d_norm_sums, vocab_size, num_batches);
        cudaDeviceSynchronize();
        cudaFree(d_nucleus);
        cudaFree(d_norm_sums);
        // step 3.3.
        float* d_blockSums = nullptr;
        cudaMalloc(&d_blockSums, norm_blocks * num_batches * sizeof(float));
        dim3 norm_grid(norm_blocks, 1, num_batches);
        cudaMalloc(&d_p_n_nucleus, max_nucleus_bytes * num_batches);
        (prefix_sum<threads><<<norm_grid, threads>>>(d_n_nucleus, d_p_n_nucleus, d_blockSums, max_nucleus_size));
        cudaDeviceSynchronize();
        cudaFree(d_n_nucleus);
        cudaFree(d_blockSums);
    });

    // // // STEP 4:  sample: d_p_n_nucleus -> sampled_token

    TIMER(samp_timer, max_nucleus_size * sizeof(float) * num_batches, num_batches * sizeof(int), max_nucleus_size * num_batches, "STEP 4: sample", {
        solve_sample(d_p_n_nucleus, seed, sampled_token, max_nucleus_size, desc_indices, vocab_size, num_batches);
        cudaDeviceSynchronize();
        cudaFree(d_p_n_nucleus);
    });

    cudaFree(desc_indices);

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