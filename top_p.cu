#include <cuda_runtime.h>
#include <vector>
#include <algorithm>
#include <stdio.h>
#include <cfloat>
#include <assert.h>
#include <iostream>
#include <stdint.h>

#define WarpsInBlock 32
#define threads (32 * WarpsInBlock)


// // // ******************************** STEP 1: SOFTMAX(LOGITS) = G_PROBS ******************************** // // //

// Kernel 1: reduce max step
template<int elemsPerThread, int THREADS>
__global__ void reduce_max_step(
    const float* __restrict__ input,
    float* __restrict__ output,
    int N
){
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int batch = blockIdx.z;
    const int tile_size = elemsPerThread * THREADS;
    const float* input_batch = input + batch * N;

    __shared__ float s[THREADS];

    int element = 0;
    float curr_max = -FLT_MAX;
    #pragma unroll
    for(int e = 0; e < elemsPerThread; ++e){
        element = bid * tile_size + e * THREADS + tid;
        if(element < N) curr_max = fmaxf(curr_max, input_batch[element]);
    }

    s[tid] = curr_max;
    __syncthreads();

    #pragma unroll
    for(int stride = THREADS >> 1; stride > 32; stride >>= 1){
        if(tid < stride) s[tid] = fmaxf(s[tid], s[tid + stride]);
        __syncthreads();
    }

    if(tid < 32){
        s[tid] = fmaxf(s[tid], s[tid + 32]);

        unsigned mask = 0xFFFFFFFF;
        float v = s[tid];
        v = fmaxf(v, __shfl_down_sync(mask, v, 16));
        v = fmaxf(v, __shfl_down_sync(mask, v, 8));
        v = fmaxf(v, __shfl_down_sync(mask, v, 4));
        v = fmaxf(v, __shfl_down_sync(mask, v, 2));
        v = fmaxf(v, __shfl_down_sync(mask, v, 1));

        if(tid == 0) {
            int out_idx = batch * ((N + tile_size - 1) / tile_size) + bid;
            output[out_idx] = v;
        }
    }
}


//Kernel 2: exp(x_i - xmax) + partial sum
template<int elemsPerThread, int THREADS>
__global__ void exp_and_partial_sum(
    const float* __restrict__ input,
    float* __restrict__ exp_output,
    float* __restrict__ output,
    int N, 
    const float* x_max
){
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int batch = blockIdx.z;
    const int tile_size = elemsPerThread * THREADS;
    const float* input_batch = input + batch * N;
    float* exp_output_batch = exp_output + batch * N;
    __shared__ float s[THREADS];
    float d_xmax = x_max[batch];

    float val = 0.0f;
    #pragma unroll
    for(int e = 0; e < elemsPerThread; ++e){
        int gid = bid * tile_size + e * THREADS + tid;
        if(gid < N){
            float exp_val = expf(input_batch[gid] - d_xmax);
            exp_output_batch[gid] = exp_val;
            val += exp_val;
        }
    }
    s[tid] = val;
    __syncthreads();

    for(int stride = THREADS >> 1; stride > 32; stride >>= 1){
        if(tid < stride) s[tid] += s[tid + stride];
        __syncthreads();
    }

    if(tid < 32){
        s[tid] += s[tid + 32];

        float v = s[tid];
        #pragma unroll
        for(int stride = 16; stride > 0; stride >>= 1) v += __shfl_down_sync(0xFFFFFFFF, v, stride);

        if(tid == 0) {
            int out_idx = batch * ((N + tile_size - 1) / tile_size) + bid;
            output[out_idx] = v;
        }
    }
}


// Kernel 3: reduce sum
template<int elemsPerThread, int THREADS>
__global__ void reduce_sum_step(
    const float* __restrict__ input,
    float* __restrict__ output,
    int N
){
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int batch = blockIdx.z;
    const int tile_size = THREADS * elemsPerThread;
    const float* input_batch = input + batch * N;

    __shared__ float s[THREADS];
    
    int element = 0;
    float sum = 0.0f;
    #pragma unroll
    for(int e = 0; e < elemsPerThread; ++e){
        element = bid * tile_size + e * THREADS + tid;
        if(element < N) sum += input_batch[element];
    }

    s[tid] = sum;
    __syncthreads();

    #pragma unroll
    for(int stride = THREADS >> 1; stride > 32; stride >>= 1){
        if(tid < stride) s[tid] += s[tid + stride];
        __syncthreads();
    }

    if(tid < 32){
        s[tid] += s[tid + 32];

        float v = s[tid];
        unsigned mask = 0xFFFFFFFF;

        v += __shfl_down_sync(mask, v, 16);
        v += __shfl_down_sync(mask, v, 8);
        v += __shfl_down_sync(mask, v, 4);
        v += __shfl_down_sync(mask, v, 2);
        v += __shfl_down_sync(mask, v, 1);

        if (tid == 0) {
            int out_idx = batch * ((N + tile_size - 1) / tile_size) + bid;
            output[out_idx] = v;
        }
    }
}


// Kernel 4: exp(x_i - xmax) / sum
template<int elemsPerThread, int THREADS>
__global__ void normalize(
    const float* __restrict__ input,
    float* __restrict__ output,
    int N, 
    const float* __restrict__ sums
){
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int batch = blockIdx.z;
    const int tile_size = elemsPerThread * THREADS;
    const float* input_batch = input + batch * N;
    float* output_batch = output + batch * N;
    float sum = sums[batch];

    #pragma unroll
    for(int e = 0; e < elemsPerThread; ++e){
        int gid = bid * tile_size + e * THREADS + tid;
        if(gid < N) output_batch[gid] = input_batch[gid] / sum;
    }
}

// // // ******************************** STEP 2: SORT(G_PROBS) = S_G_PROBS ******************************** // // //

template<int THREADS>
__global__ void init_indices(
    unsigned int* __restrict__ indices,
    int N
){
    int gid = blockIdx.x * THREADS + threadIdx.x;
    int batch = blockIdx.z;
    unsigned int* indices_batch = indices + batch * N;
    if (gid < N) {
        indices_batch[gid] = gid;
    }
}

// Convert probabilities in [0,1] to 32-bit unsigned integer keys for radix sort
// Scales to [0, UINT32_MAX] preserving order; clamped to [0,1].
template<int THREADS>
__global__ void fp32_to_uint32_kernel(
    const float* __restrict__ input,
    unsigned int* __restrict__ output,
    int N
){
    int gid = blockIdx.x * THREADS + threadIdx.x;
    int batch = blockIdx.z;
    if (gid >= N) return;
    const float* input_batch = input + batch * N;
    unsigned int* output_batch = output + batch * N;
    float v = input_batch[gid];
    // clamp to [0,1]
    v = fminf(fmaxf(v, 0.0f), 1.0f);
    // scale to 32-bit range and round to nearest
    unsigned int key = __float2uint_rn(v * 4294967295.0f);
    output_batch[gid] = key;
}

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

// // // **** BITONIC SORT **** // // //
// TODO: reverse sorting so output is descending + track token indices
// template<int THREADS>
// __global__ void fill_the_end(
//     unsigned int* input, int N, int M, int val    
// ){
//     const int gid = blockIdx.x * THREADS + threadIdx.x;
//     if(gid>= N && gid < M) input[gid] = val;
// }

// template<int THREADS>
// __global__ void bitonic_sort_step(unsigned int* input, int j, int k, int N){

//     int i = blockIdx.x * THREADS + threadIdx.x;
//     int ixj = i ^ j;
//     if(ixj <= i) return;

//     bool ascending = ((i & k) == 0);

//     float a = input[i];
//     float b = input[ixj];

//     if( (ascending && a > b) || (!ascending && a < b) ){
//         input[i] = b;
//         input[ixj] = a;
//     }
// }

// extern "C" void solve_bitonic_sort(unsigned int* input, unsigned int* output, int N){
//     int M = 1;
//     while(M < N) M <<= 1;

//     unsigned int* padded = nullptr;
//     cudaMalloc(&padded, M * sizeof(float));
//     cudaMemcpy(padded, input, N * sizeof(float), cudaMemcpyDeviceToDevice);
    
//     const int blocks = (M + threads - 1) / threads;

//     fill_the_end<threads><<<blocks, threads>>>(padded, N, M, 4294967295);

//     for(int k = 2; k < M; k <<= 1){
//         for(int j = k >> 1; j > 0; j >>= 1){
//             bitonic_sort_step<threads><<<blocks, threads>>>(padded, j, k, M);
//             cudaDeviceSynchronize();
//         }
//     }

//     cudaMemcpy(output, padded, N * sizeof(unsigned int), cudaMemcpyDeviceToDevice);
//     cudaFree(padded);
// }

// // // **** RADIX SORT **** // // //


// Binary (base 2) LSD radix

// per warp prefix sum
// val = per-thread starting value
__device__ __forceinline__ int prefix_per_warp(int val) {
    int lane = threadIdx.x % 32;
    for (int offset = 16; offset > 0; offset >>= 1) {
        int v = __shfl_up_sync(0xffffffff, val, offset);
        if (lane >= offset) {
            val += v;
        }

    }
    return val;
}

// per block prefix sum
// val = per-thread starting value
__device__ __forceinline__ int prefix_per_block_helper(int val, int& block_sum) {

    int tid = threadIdx.x;
    int warp = tid / 32;
    int lane = tid % 32;

    int sum = prefix_per_warp(val);
    __shared__ int smem[WarpsInBlock];
    if (lane == 31) {
        smem[warp] = sum;
    }
    __syncthreads();

    if (warp == 0) {
        val = (lane < WarpsInBlock) ? smem[lane] : 0;
        int block_prefix_sum = prefix_per_warp(val);
        smem[lane] = block_prefix_sum - val;            // sum of all previous warp sums - used below in return
        if (lane == WarpsInBlock - 1) {
            block_sum = block_prefix_sum;               // sum of all warp sums = block sum
        }
    }
    __syncthreads();

    return sum + smem[warp];                            // convert each thread's warp-level prefix into block-level prefix
}

template<int THREADS>
__global__ void prefix_per_block_desc(
    unsigned int *input, 
    unsigned int *block_sums, 
    int bit,
    int N
) {
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int batch = blockIdx.z;

    unsigned int* input_batch = input + batch * N;
    int gid = bid * THREADS + tid;
    int val = 0;

    if (gid < N) {
        // For ascending order: count 0-bits (low values first)
        val = 1 - ((input_batch[gid] >> bit) & 1);
    }

    int block_sum = 0;
    val = prefix_per_block_helper(val, block_sum);

    if (tid == WarpsInBlock - 1) {
        int out_idx = batch * ((N + THREADS - 1) / THREADS) + bid;
        block_sums[out_idx] = block_sum;
    }
}

template<int THREADS>
__global__ void radix_sort_asc_kernel(
    unsigned int *input, 
    unsigned int *output,
    unsigned int *input_indices,
    unsigned int *output_indices,
    unsigned int *offsets,
    unsigned int *total_ones,
    int bit,
    int N
) {
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int batch = blockIdx.z;
    unsigned int* input_batch = input + batch * N;
    unsigned int* output_batch = output + batch * N;
    unsigned int* input_indices_batch = input_indices + batch * N;
    unsigned int* output_indices_batch = output_indices + batch * N;
    int gid = bid * THREADS + tid;
    unsigned int tz = total_ones[batch];

    int val = 0;
    if (gid < N) {
        // Count 0-bits (low values first for ascending)
        val = 1 - ((input_batch[gid] >> bit) & 1);
    }
    
    int block_sum = 0;
    int prefix_in_block = prefix_per_block_helper(val, block_sum);

    int blocks_per_batch = (N + THREADS - 1) / THREADS;
    int offset = prefix_in_block + offsets[batch * blocks_per_batch + bid];

    if (gid < N) {
        unsigned int a = input_batch[gid];
        int is_one = (a >> bit) & 1;
        int idx;
        
        if (is_one) {
            // 0-bits (low values) go at the beginning: offsets[bid] + prefix_in_block - 1
            idx = tz + gid - offset;
        } else {
            // 1-bits (high values) go after all 0s: total_zeros + gid - offsets[bid] - prefix_in_block
            idx = offset - 1;
        }
        
        output_batch[idx] = a;
        output_indices_batch[idx] = input_indices_batch[gid];
    }
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

// // // ******************************** STEP 3: NUCLEUS(S_G_PROBS) = NUCLEUS ******************************** // // //

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

    constexpr int elemsPerThread = 16;
    
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

    // // // // // 6 STEPS:
    // // // STEP 1: softmax(logits) = g_probs
    // // // STEP 2 sort(global_probs) = s_g_probs
    // // // STEP 3.1. nucleus(s_g_probs) = nucleus [via prefix sum]
    // // // STEP 3.2. normalize(nucleus) = n_nucleus
    // // // STEP 3.3. prefix_sum(n_nucleus) = p_n_nucleus
    // // // STEP 4: sample(p_n_nucleus) = sampled_token

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

    cudaEventRecord(start);

    float* g_probs = nullptr;
    cudaMalloc(&g_probs, vocab_size * num_batches * sizeof(float));
    
    const int tile_size = elemsPerThread * threads;
    const int first_blocks = (vocab_size + tile_size - 1) / tile_size;
    dim3 first_grid(first_blocks, 1, num_batches);

    // // Kernel 1: reduce max
    float *bufA = nullptr;
    float *bufB = nullptr;
    cudaMalloc(&bufA, first_blocks * num_batches * sizeof(float));
    cudaMalloc(&bufB, first_blocks * num_batches * sizeof(float));

    reduce_max_step<elemsPerThread, threads><<<first_grid, threads>>>(logits, bufA, vocab_size);

    float* src = bufA;
    float* dst = bufB;

    int curr_size = first_blocks;

    while(curr_size > 1){
        int curr_blocks = (curr_size + tile_size - 1) / tile_size;
        dim3 curr_grid(curr_blocks, 1, num_batches);
        reduce_max_step<elemsPerThread, threads><<<curr_grid, threads>>>(src, dst, curr_size);
        curr_size = curr_blocks;
        float *tmp = dst; dst = src; src = tmp;
    }
    //(src now points to per-batch global max)
    float *d_xmax = src;

    cudaDeviceSynchronize();

    // // Kernel 2: exp(x_i - xmax) + partial sum
    float* exp_output = nullptr;
    float* partial_sum = nullptr;
    cudaMalloc(&exp_output, vocab_size * num_batches * sizeof(float));
    cudaMalloc(&partial_sum, first_blocks * num_batches * sizeof(float));
    exp_and_partial_sum<elemsPerThread, threads><<<first_grid, threads>>>(logits, exp_output, partial_sum, vocab_size, d_xmax);
    //( exp_output now points to array of exp(x_i - xmax) )

    cudaDeviceSynchronize();

    // // Kernel 3: reduce sum

    // start reduction from per-block partial sums (written by exp_and_partial_sum)
    float* src_sum = partial_sum;
    float* dst_sum = bufA;
    curr_size = first_blocks;

    while(curr_size > 1){
        int curr_blocks = (curr_size + tile_size - 1) / tile_size;
        dim3 curr_grid(curr_blocks, 1, num_batches);
        reduce_sum_step<elemsPerThread, threads><<<curr_grid, threads>>>(src_sum, dst_sum, curr_size);
        curr_size = curr_blocks;
        // swap pointers so dst becomes next src and vice versa
        float* tmp = dst_sum; dst_sum = src_sum; src_sum = tmp;
    }
    //(src_sum now points to per-batch global sum of exp(x_i - xmax))
    float* d_exp_sum = src_sum;

    // // Kernel 4: exp(x_i - xmax) / sum
    normalize<elemsPerThread, threads><<<first_grid, threads>>>(exp_output, g_probs, vocab_size, d_exp_sum);

    cudaFree(bufA);
    cudaFree(bufB);
    cudaFree(exp_output);
    cudaFree(partial_sum);
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float time_step1 = 0.0f;
    cudaEventElapsedTime(&time_step1, start, stop);
    long long bytes_step1 = vocab_size * sizeof(float) * 2; // read logits, write g_probs
    long long flops_step1 = vocab_size * 10; // approximate: exp, div, reduce ops
    timings.push_back({"STEP 1: Softmax", time_step1, bytes_step1, (long long)((long long)vocab_size * sizeof(float)), flops_step1});

    int blocksx = (vocab_size + threads - 1) / threads;
    dim3 grid(blocksx, 1, num_batches);

    // // // STEP 2: sort(g_probs) = s_g_probs

    cudaEventRecord(start);

    // Radix/Bitonic sort requires uints as inputs
    unsigned int* uint32_g_probs = nullptr;
    cudaMalloc(&uint32_g_probs, vocab_size * num_batches * sizeof(unsigned int));
    fp32_to_uint32_kernel<threads><<<grid, threads>>>(g_probs, uint32_g_probs, vocab_size);
    cudaDeviceSynchronize();
    cudaFree(g_probs);

    unsigned int *d_in, *d_out, *d_indices_in, *d_indices_out;
    cudaMalloc(&d_in, vocab_size * num_batches * sizeof(unsigned int));
    cudaMalloc(&d_out, vocab_size * num_batches * sizeof(unsigned int));
    cudaMalloc(&d_indices_in, vocab_size * num_batches * sizeof(unsigned int));
    cudaMalloc(&d_indices_out, vocab_size * num_batches * sizeof(unsigned int));

    cudaMemcpy(d_in, uint32_g_probs, vocab_size * num_batches * sizeof(unsigned int), cudaMemcpyDeviceToDevice);
    init_indices<threads><<<grid, threads>>>(d_indices_in, vocab_size);
    init_indices<threads><<<grid, threads>>>(d_indices_out, vocab_size);

    std::vector<unsigned int> h_block_sums(blocksx);
    std::vector<unsigned int> h_offsets(blocksx);

    unsigned int *d_block_sums;
    cudaMalloc(&d_block_sums, blocksx * num_batches * sizeof(unsigned int));

    // Store total_ones per batch for each bit iteration
    unsigned int *d_total_ones;
    cudaMalloc(&d_total_ones, 32 * num_batches * sizeof(unsigned int));

    cudaEventRecord(start);
    for (int iter = 0; iter < 32; iter++) {

        // Compute prefix sums for all batches at once
        prefix_per_block_desc<threads><<<grid, threads>>>(d_in, d_block_sums, iter, vocab_size);

        // Compute total_ones per batch on host (still necessary for correctness)
        std::vector<unsigned int> h_total_ones(num_batches, 0);
        for(int b = 0; b < num_batches; ++b){
            cudaMemcpy(h_block_sums.data(), d_block_sums + b * blocksx, blocksx * sizeof(unsigned int), cudaMemcpyDeviceToHost);
            
            unsigned int total_ones = 0;
            for (int i = 0; i < blocksx; i++) {
                h_offsets[i] = total_ones;
                total_ones += h_block_sums[i];
            }
            h_total_ones[b] = total_ones;
            cudaMemcpy(d_block_sums + b * blocksx, h_offsets.data(), blocksx * sizeof(unsigned int), cudaMemcpyHostToDevice);
        }
        
        // Copy total_ones to device and launch sort for all batches
        cudaMemcpy(d_total_ones + iter * num_batches, h_total_ones.data(), num_batches * sizeof(unsigned int), cudaMemcpyHostToDevice);
        
        radix_sort_asc_kernel<threads><<<grid, threads>>>(
            d_in, d_out, 
            d_indices_in, d_indices_out, 
            d_block_sums, 
            d_total_ones + iter * num_batches, iter, vocab_size);

        std::swap(d_in, d_out);
        std::swap(d_indices_in, d_indices_out);
    }
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float time_sort = 0.0f;
    cudaEventElapsedTime(&time_sort, start, stop);
    long long bytes_sort = vocab_size * sizeof(unsigned int) * 2; // read d_in, write d_out
    long long flops_sort = vocab_size * 32; // 32-bit sort
    timings.push_back({"STEP 2.1: Sort", time_sort, bytes_sort, (long long)(vocab_size * sizeof(unsigned int)), flops_sort});
    
    cudaFree(d_total_ones);

    // Reverse to get descending order - all batches at once
    cudaEventRecord(start);
    int rev_blocks = (vocab_size + threads - 1) / threads;
    dim3 rev_grid(rev_blocks, 1, num_batches);
    reverse_array<threads><<<rev_grid, threads>>>(d_in, d_indices_in, vocab_size);
    
    float* s_g_probs = nullptr;
    cudaMalloc(&s_g_probs, vocab_size * num_batches * sizeof(float));
    
    //convert back to float
    uint32_to_fp32_kernel<threads><<<grid, threads>>>(d_in, s_g_probs, vocab_size);
    cudaDeviceSynchronize();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float time_reverse = 0.0f;
    cudaEventElapsedTime(&time_reverse, start, stop);
    long long bytes_reverse = vocab_size * sizeof(unsigned int) * 2 + vocab_size * sizeof(float);
    long long flops_reverse = vocab_size;
    timings.push_back({"STEP 2.2: Reverse", time_reverse, bytes_reverse, (long long)(vocab_size * sizeof(float)), flops_reverse});

    cudaFree(d_in);
    cudaFree(d_out);
    cudaFree(d_indices_out);
    cudaFree(d_block_sums);


    // // // STEP 3: Nucleus

    // NOTE: We loop over batches here (exception to rule 4) because each batch can have different nucleus size
    // Allocate per-batch tracking arrays
    std::vector<int> h_nucleus_sizes(num_batches);
    std::vector<float> h_global_probs_sums(num_batches);
    std::vector<unsigned int*> d_nucleus_arrays(num_batches);
    std::vector<float*> d_n_nucleus_arrays(num_batches);
    std::vector<float*> d_p_n_nucleus_arrays(num_batches);
    
    int max_nucleus_size = 0;
    
    cudaEventRecord(start);
    
    // STEP 3.1: Find nucleus size via prefix sum on sorted probabilities

    for(int batch = 0; batch < num_batches; ++batch) {
        // STEP 3.1: Find nucleus size via prefix sum on sorted probabilities
        float* d_blockSums = reinterpret_cast<float*>(d_block_sums); // reuse buffer
        cudaMemset(d_blockSums, 0, blocksx * sizeof(float));
        
        float* d_per_block_sums = nullptr;
        cudaMalloc(&d_per_block_sums, sizeof(float) * vocab_size);
        
        // Prefix sum on batch's sorted probabilities
        const float* batch_s_g_probs = s_g_probs + batch * vocab_size;
        prefix_sum<threads><<<blocksx, threads>>>(batch_s_g_probs, d_per_block_sums, d_blockSums, vocab_size);
        
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
        cudaMemcpy(h_sorted_probs.data(), s_g_probs + batch * vocab_size, nucleus_size * sizeof(float), cudaMemcpyDeviceToHost);
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
    float* d_nucleus_alloc = nullptr;
    float* d_n_nucleus_alloc = nullptr;
    float* d_p_n_nucleus_alloc = nullptr;
    int* nucleus_sampled_token_alloc = nullptr;
    
    cudaMalloc(&d_nucleus_alloc, max_nucleus_bytes * num_batches);
    cudaMalloc(&d_n_nucleus_alloc, max_nucleus_bytes * num_batches);
    cudaMalloc(&d_p_n_nucleus_alloc, max_nucleus_bytes * num_batches);
    cudaMalloc(&nucleus_sampled_token_alloc, num_batches * sizeof(int));
    
    // Zero-initialize nucleus allocation to handle padding for batches with nucleus_size < max_nucleus_size
    cudaMemset(d_nucleus_alloc, 0, max_nucleus_bytes * num_batches);
    
    // Copy nucleus portions from sorted probs for all batches
    cudaDeviceSynchronize();  // Ensure conversion is complete
    
    // Copy nucleus portions from sorted probs to nucleus allocation for all batches
    for(int batch = 0; batch < num_batches; ++batch) {
        int nucleus_size = h_nucleus_sizes[batch];
        int nucleus_bytes = nucleus_size * sizeof(float);
        
        // Copy from sorted probs device buffer to nucleus allocation (device-to-device)
        cudaMemcpy(d_nucleus_alloc + batch * max_nucleus_size, 
                   s_g_probs + batch * vocab_size, 
                   nucleus_bytes, cudaMemcpyDeviceToDevice);
    }
    
    // STEP 3.2: Normalize
    int norm_blocks = (max_nucleus_size + final_threads - 1) / final_threads;
    dim3 norm_grid(norm_blocks, 1, num_batches);
    
    // Need to create per-batch sums array for normalize kernel
    float* d_norm_sums = nullptr;
    cudaMalloc(&d_norm_sums, num_batches * sizeof(float));
    cudaMemcpy(d_norm_sums, h_global_probs_sums.data(), num_batches * sizeof(float), cudaMemcpyHostToDevice);
    
    cudaEventRecord(start);
    normalize<elemsPerThread, final_threads><<<norm_grid, final_threads>>>(d_nucleus_alloc, d_n_nucleus_alloc, max_nucleus_size, d_norm_sums);
    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float time_norm = 0.0f;
    cudaEventElapsedTime(&time_norm, start, stop);
    long long bytes_norm = max_nucleus_size * sizeof(float) * 2 * num_batches; // read nucleus, write normalized
    long long flops_norm = max_nucleus_size * num_batches; // division per element
    timings.push_back({"STEP 3.2: Norm", time_norm, bytes_norm, (long long)(max_nucleus_size * sizeof(float) * num_batches), flops_norm});
    
    // STEP 3.3: Prefix sum for all batches
    float* d_blockSums_alloc = nullptr;
    cudaMalloc(&d_blockSums_alloc, norm_blocks * num_batches * sizeof(float));
    
    cudaEventRecord(start);
    prefix_sum<final_threads><<<norm_grid, final_threads>>>(d_n_nucleus_alloc, d_p_n_nucleus_alloc, d_blockSums_alloc, max_nucleus_size);
    cudaDeviceSynchronize();
    
    // STEP 3.4: Sample for all batches
    dim3 sample_grid(1, 1, num_batches);
    warp_sample<<<sample_grid, final_threads>>>(d_p_n_nucleus_alloc, seed, nucleus_sampled_token_alloc, max_nucleus_size);
    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float time_samp = 0.0f;
    cudaEventElapsedTime(&time_samp, start, stop);
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
        const unsigned int* batch_d_indices_in = d_indices_in + batch * vocab_size;
        cudaMemcpy(&original_token, batch_d_indices_in + sampled_nucleus_idx, sizeof(unsigned int), cudaMemcpyDeviceToHost);
        
        cudaMemcpy(sampled_token + batch, &original_token, sizeof(unsigned int), cudaMemcpyHostToDevice);
    }
    
    // Cleanup
    cudaFree(d_nucleus_alloc);
    cudaFree(d_n_nucleus_alloc);
    cudaFree(d_p_n_nucleus_alloc);
    cudaFree(d_blockSums_alloc);
    cudaFree(nucleus_sampled_token_alloc);
    cudaFree(d_norm_sums);

    // Cleanup
    cudaFree(s_g_probs);
    cudaFree(uint32_g_probs);
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
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

}