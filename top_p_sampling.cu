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
    const int tile_size = elemsPerThread * THREADS;

    __shared__ float s[THREADS];

    int element = 0;
    float curr_max = -FLT_MAX;
    #pragma unroll
    for(int e = 0; e < elemsPerThread; ++e){
        element = bid * tile_size + e * THREADS + tid;
        if(element < N) curr_max = fmaxf(curr_max, input[element]);
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

        if(tid == 0) output[bid] = v;
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
    const int tile_size = elemsPerThread * THREADS;
    __shared__ float s[THREADS];
    float d_xmax = *x_max;

    float val = 0.0f;
    #pragma unroll
    for(int e = 0; e < elemsPerThread; ++e){
        int gid = bid * tile_size + e * THREADS + tid;
        if(gid < N){
            float exp_val = expf(input[gid] - d_xmax);
            exp_output[gid] = exp_val;
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

        if(tid == 0) output[bid] = v;
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
    const int tile_size = THREADS * elemsPerThread;

    __shared__ float s[THREADS];
    
    int element = 0;
    float sum = 0.0f;
    #pragma unroll
    for(int e = 0; e < elemsPerThread; ++e){
        element = bid * tile_size + e * THREADS + tid;
        if(element < N) sum += input[element];
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

        if (tid == 0) output[bid] = v;
    }
}


// Kernel 4: exp(x_i - xmax) / sum
template<int elemsPerThread, int THREADS>
__global__ void normalize(
    const float* __restrict__ input,
    float* __restrict__ output,
    int N, 
    float sum
){
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int tile_size = elemsPerThread * THREADS;

    #pragma unroll
    for(int e = 0; e < elemsPerThread; ++e){
        int gid = bid * tile_size + e * THREADS + tid;
        if(gid < N) output[gid] = input[gid] / sum;
    }
}

// // // ******************************** STEP 2: SORT(G_PROBS) = S_G_PROBS ******************************** // // //

__global__ void init_indices(
    unsigned int* __restrict__ indices,
    int N
){
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < N) {
        indices[gid] = gid;
    }
}

// Convert probabilities in [0,1] to 32-bit unsigned integer keys for radix sort
// Scales to [0, UINT32_MAX] preserving order; clamped to [0,1].
__global__ void fp32_to_uint32_kernel(
    const float* __restrict__ input,
    unsigned int* __restrict__ output,
    int N
){
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= N) return;
    float v = input[gid];
    // clamp to [0,1]
    v = fminf(fmaxf(v, 0.0f), 1.0f);
    // scale to 32-bit range and round to nearest
    unsigned int key = __float2uint_rn(v * 4294967295.0f);
    output[gid] = key;
}

__global__ void uint32_to_fp32_kernel(
    const unsigned int* __restrict__ input,
    float* __restrict__ output,
    int N
){
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if(gid >= N) return;
    unsigned int i = input[gid];
    // convert uint32 in [0, 2^32-1] back to float in [0,1]
    float v = __uint2float_rn(i) / 4294967295.0f;
    output[gid] = v;
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

__global__ void prefix_per_block_desc(
    unsigned int *input, 
    unsigned int *block_sums, 
    int bit,
    int N
) {
    int tid = threadIdx.x;
    int bid = blockIdx.x;

    int gid = bid * blockDim.x + tid;
    int val = 0;

    if (gid < N) {
        // For ascending order: count 0-bits (low values first)
        val = 1 - ((input[gid] >> bit) & 1);
    }

    int block_sum = 0;
    val = prefix_per_block_helper(val, block_sum);

    if (tid == WarpsInBlock - 1) {
        block_sums[bid] = block_sum;
    }
}

__global__ void radix_sort_asc_kernel(
    unsigned int *input, 
    unsigned int *output,
    unsigned int *input_indices,
    unsigned int *output_indices,
    unsigned int *offsets,
    unsigned int total_zeros,
    int bit,
    int N
) {
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int gid = bid * blockDim.x + tid;

    int val = 0;
    if (gid < N) {
        // Count 0-bits (low values first for ascending)
        val = 1 - ((input[gid] >> bit) & 1);
    }
    
    int block_sum = 0;
    int prefix_in_block = prefix_per_block_helper(val, block_sum);

    int offset = prefix_in_block + offsets[bid];

    if (gid < N) {
        unsigned int a = input[gid];
        int is_one = (a >> bit) & 1;
        int idx;
        
        if (is_one) {
            // 0-bits (low values) go at the beginning: offsets[bid] + prefix_in_block - 1
            idx = total_zeros + gid - offset;
        } else {
            // 1-bits (high values) go after all 0s: total_zeros + gid - offsets[bid] - prefix_in_block
            idx = offset - 1;
        }
        
        output[idx] = a;
        output_indices[idx] = input_indices[gid];
    }
}

__global__ void reverse_array(
    unsigned int* input,
    unsigned int* input_indices,
    int N
){
    const int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int halfN = (N + 1) / 2;
    if(gid < halfN){
        unsigned int left_element = input[gid];
        unsigned int left_idx = input_indices[gid];
        unsigned int right_element = input[N - 1 - gid];
        unsigned int right_idx = input_indices[N - 1 - gid];
        input[gid] = right_element;
        input_indices[gid] = right_idx;
        input[N - 1 - gid] = left_element;
        input_indices[N - 1 - gid] = left_idx;
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
    // assert( N % THREADS == 0 );
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int block_start = bid * THREADS;
    int gid = block_start + tid;

    __shared__ float s[THREADS];
    if(gid < N){
        s[tid] = input[gid];
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

    if(gid < N) output[gid] = s[tid];
    __syncthreads();

    if(tid == 0) blockSums[bid] = s[THREADS - 1];

}

template<int THREADS>
__global__ void add_block_offset(
    const float* __restrict__ input,
    const float* __restrict__ offset,
    float* __restrict__ output,
    int N
){  
    int bid = blockIdx.x;
    int gid = bid * THREADS + threadIdx.x;
    if(gid >= N) return;
    float off = (bid == 0) ? 0.0f : offset[bid - 1];
    output[gid] = input[gid] + off;
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
    int lane = threadIdx.x % 32;

    float sampled_prob = lcg_uniform(*seed);
    // printf("sampled_prob: %f\n", sampled_prob);


    for(int i = 0; i < vocab_size; i += 32){
        int idx = i + lane;
        float val = 1.1f;
        if(idx < vocab_size) val = prefix_sum_probs[idx];
        uint32_t mask = __ballot_sync(0xFFFFFFFFu, val >= sampled_prob);

        if (mask != 0){
            //__ffs() = find first set = return 1-based indexed of the first 1 in the integer mask
            if (lane == (__ffs(mask) - 1)) {
                // printf("selected idx: %d\n", idx);
                *sampled_token = idx;
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
    float* sampled_prob = nullptr
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
        long long peak_memory;
        float h2d_time_ms;
        
        TimingResult(const char* n, float t, long long br, long long bw, long long f, long long pm = 0, float h2d = 0) 
            : name(n), time_ms(t), bytes_read(br), bytes_written(bw), flops(f), peak_memory(pm), h2d_time_ms(h2d) {}
        
        void print() const {
            float mb_read = bytes_read / (1024.0f * 1024);
            float mb_write = bytes_written / (1024.0f * 1024);
            float gbs_read = (bytes_read / (1024.0f * 1024 * 1024)) / (time_ms / 1000.0f);
            float gbs_write = (bytes_written / (1024.0f * 1024 * 1024)) / (time_ms / 1000.0f);
            float gflops = (flops / 1e9f) / (time_ms / 1000.0f);
            float total_bytes = bytes_read + bytes_written;
            float arith_intensity = (total_bytes > 0) ? ((flops / 1e9f) / (total_bytes / (1024.0f * 1024 * 1024))) : 0.0f;
            
            printf("%-20s | %9.3f | %9.3f | %11.1f | %10.3f | %12.1f | %6.1f | %7.3f\n",
                   name, time_ms, mb_read, gbs_read, mb_write, gbs_write, gflops, arith_intensity);
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
    cudaMalloc(&g_probs, vocab_size * sizeof(float));
    
    const int tile_size = elemsPerThread * threads;
    const int first_blocks = (vocab_size + tile_size - 1) / tile_size;

    // // Kernel 1: reduce max
    float *bufA = nullptr;
    float *bufB = nullptr;
    cudaMalloc(&bufA, first_blocks * sizeof(float));
    cudaMalloc(&bufB, first_blocks * sizeof(float));

    reduce_max_step<elemsPerThread, threads><<<first_blocks, threads>>>(logits, bufA, vocab_size);

    float* src = bufA;
    float* dst = bufB;

    int curr_size = first_blocks;

    while(curr_size > 1){
        int curr_blocks = (curr_size + tile_size - 1) / tile_size;
        reduce_max_step<elemsPerThread, threads><<<curr_blocks, threads>>>(src, dst, curr_size);
        curr_size = curr_blocks;
        float *tmp = dst; dst = src; src = tmp;
    }
    //(src now points to global max)
    float *d_xmax = src;

    cudaDeviceSynchronize();

    // // Kernel 2: exp(x_i - xmax) + partial sum
    float* exp_output = nullptr;
    float* partial_sum = nullptr;
    cudaMalloc(&exp_output, vocab_size * sizeof(float));
    cudaMalloc(&partial_sum, first_blocks * sizeof(float));
    exp_and_partial_sum<elemsPerThread, threads><<<first_blocks, threads>>>(logits, exp_output, partial_sum, vocab_size, d_xmax);
    //( exp_output now points to array of exp(x_i - xmax) )

    cudaDeviceSynchronize();

    // // Kernel 3: reduce sum

    // start reduction from per-block partial sums (written by exp_and_partial_sum)
    float* src_sum = partial_sum;
    float* dst_sum = bufA;
    curr_size = first_blocks;

    while(curr_size > 1){
        int curr_blocks = (curr_size + tile_size - 1) / tile_size;
        reduce_sum_step<elemsPerThread, threads><<<curr_blocks, threads>>>(src_sum, dst_sum, curr_size);
        curr_size = curr_blocks;
        // swap pointers so dst becomes next src and vice versa
        float* tmp = dst_sum; dst_sum = src_sum; src_sum = tmp;
    }
    //(src_sum now points to global sum of exp(x_i - xmax))
    float exp_sum = 0.0f;
    cudaMemcpy(&exp_sum, src_sum, sizeof(float), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    // // Kernel 4: exp(x_i - xmax) / sum
    normalize<elemsPerThread, threads><<<first_blocks, threads>>>(exp_output, g_probs, vocab_size, exp_sum);

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

    

    // // // STEP 2: sort(g_probs) = s_g_probs
    cudaEventRecord(start);

    // Radix/Bitonic sort requires uints as inputs
    unsigned int* uint32_g_probs = nullptr;
    cudaMalloc(&uint32_g_probs, vocab_size * sizeof(unsigned int));
    fp32_to_uint32_kernel<<<first_blocks, threads>>>(g_probs, uint32_g_probs, vocab_size);
    cudaDeviceSynchronize();
    cudaFree(g_probs);

   int blocks = (vocab_size + threads - 1) / threads;

    unsigned int *d_in, *d_out, *d_indices_in, *d_indices_out;
    cudaMalloc(&d_in, vocab_size * sizeof(unsigned int));
    cudaMalloc(&d_out, vocab_size * sizeof(unsigned int));
    cudaMalloc(&d_indices_in, vocab_size * sizeof(unsigned int));
    cudaMalloc(&d_indices_out, vocab_size * sizeof(unsigned int));

    cudaMemcpy(d_in, uint32_g_probs, vocab_size * sizeof(unsigned int), cudaMemcpyDeviceToDevice);
    init_indices<<<first_blocks, threads>>>(d_indices_in, vocab_size);
    init_indices<<<first_blocks, threads>>>(d_indices_out, vocab_size);

    std::vector<unsigned int> h_block_sums(blocks);
    std::vector<unsigned int> h_offsets(blocks);

    unsigned int *d_block_sums;
    cudaMalloc(&d_block_sums, blocks * sizeof(unsigned int));

    for (int iter = 0; iter < 32; iter++) {

        // let ib = iter-th bit of given element e.g. 2-nd bit of 5 (= 0101) is 1.

        // I. 
        // (i) each thread in block calculates inclusive prefix (number of 1s in ib up to this thread)
        // (ii) the block calculates number of 1s in block (number of 1s in ib for all threads) into d_block_sums
        prefix_per_block_desc<<<blocks, threads>>>(d_in, d_block_sums, iter, vocab_size);

        // II. On-host exclusive scan of d_block_sums to produce per-block global offsets
        cudaMemcpy(h_block_sums.data(), d_block_sums, blocks * sizeof(unsigned int), cudaMemcpyDeviceToHost);
        int g_offset = 0;
        for (int i = 0; i < blocks; i++) {
            h_offsets[i] = g_offset;
            g_offset += h_block_sums[i];
        }
        cudaMemcpy(d_block_sums, h_offsets.data(), blocks * sizeof(unsigned int), cudaMemcpyHostToDevice);

        // III. Write the thread's element to output index:
        // (i) offsets[bid] + prefix_in_block - 1 for 0-bits (low values first for ascending)
        // (ii) total_zeros + gid - offsets[bid] - prefix_in_block for 1-bits
        radix_sort_asc_kernel<<<blocks, threads>>>(d_in, d_out, d_indices_in, d_indices_out, d_block_sums, g_offset, iter, vocab_size);

        // swap so the next iteration runs on the newly-paritioned array
        std::swap(d_in, d_out);
        std::swap(d_indices_in, d_indices_out);
    }
    //(d_in points to the result of sorting in ascending order)
    //(d_indices_in points to the corresponding sorted indices)

    // Reverse to get descending order
    int rev_blocks = (vocab_size + threads - 1) / threads;
    reverse_array<<<rev_blocks, threads>>>(d_in, d_indices_in, vocab_size);
    cudaDeviceSynchronize();
    
    //(d_in now points to the result of sorting in descending order)
    //(d_indices_in points to the corresponding sorted indices in descending order)

    float* s_g_probs = nullptr;
    cudaMalloc(&s_g_probs, vocab_size * sizeof(float));
    
    //convert back to float
    uint32_to_fp32_kernel<<<first_blocks, threads>>>(d_in, s_g_probs, vocab_size);

    cudaFree(d_in);
    cudaFree(d_out);
    cudaFree(d_indices_out);
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float time_step2 = 0.0f;
    cudaEventElapsedTime(&time_step2, start, stop);
    long long bytes_step2 = vocab_size * sizeof(float) * 2 + vocab_size * sizeof(unsigned int) * 2;
    long long flops_step2 = vocab_size * 32; // radix sort: 32 bits * comparisons per element
    timings.push_back({"STEP 2: Sort", time_step2, bytes_step2, (long long)((long long)vocab_size * sizeof(float) + (long long)vocab_size * sizeof(unsigned int)), flops_step2});

    // // // STEP 3.1.: nucleus(s_g_probs) = nucleus [via prefix sum]
    cudaEventRecord(start);

    float* d_blockSums = reinterpret_cast<float*>(d_block_sums); // reuse previous buffer
    cudaMemset(d_block_sums, 0, blocks * sizeof(float));

    float* d_per_block_sums = nullptr;
    cudaMalloc(&d_per_block_sums, sizeof(float) * vocab_size);

    // 1) per-block local prefix and block sums
    prefix_sum<threads><<<blocks, threads>>>(s_g_probs, d_per_block_sums, d_blockSums, vocab_size);

    // 2) copy block sums to host and find first block for which cumulative sum >= p
    std::vector<float> h_blockSums(blocks);
    cudaMemcpy(h_blockSums.data(), d_blockSums, sizeof(float)*blocks, cudaMemcpyDeviceToHost);

    float p_val = 0.0f;
    cudaMemcpy(&p_val, p, sizeof(float), cudaMemcpyDeviceToHost);

    int target_block = -1;
    int target_index = -1;
    float running = 0.0f;
    for (int b = 0; b < blocks; ++b) {
        running += h_blockSums[b];
        if (running >= p_val) { target_block = b; break; }
    }
    
    // If target_block == -1, the entire vocabulary is needed (nucleus encompasses all)
    bool nucleus_is_entire_vocab = (target_block == -1);
    if (nucleus_is_entire_vocab) {
        target_block = blocks - 1;
        target_index = vocab_size - 1;
    }


    // prefix sum before the target block to find the gid
    float prev_block_prefix = running - h_blockSums[target_block];

    // copy only the target block's local prefixes and search within it
    int blockStart = target_block * threads;
    int elements_in_block = min(vocab_size - blockStart, threads);      //TODO: remove later after introducing (static) asserts
    std::vector<float> h_targetBlockPrefix(elements_in_block);
    cudaMemcpy(h_targetBlockPrefix.data(), d_per_block_sums + blockStart, sizeof(float) * elements_in_block, cudaMemcpyDeviceToHost);

    float need = p_val - prev_block_prefix;
    //on-host bin search
    auto it = std::lower_bound(h_targetBlockPrefix.begin(), h_targetBlockPrefix.end(), need);
    int local_idx = (it == h_targetBlockPrefix.end()) ? (elements_in_block - 1) : int(it - h_targetBlockPrefix.begin());
    target_index = blockStart + local_idx;
    float global_probs_sum = prev_block_prefix + h_targetBlockPrefix[local_idx]; // sum of probabilities from 0 to target_index

    cudaFree(d_blockSums);

    float* nucleus = nullptr;
    int nucleus_size = (target_index + 1);
    int nucleus_bytes = nucleus_size * sizeof(float);
    cudaMalloc(&nucleus, nucleus_bytes);
    cudaMemcpy(nucleus, s_g_probs, nucleus_bytes, cudaMemcpyDeviceToDevice);
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float time_step31 = 0.0f;
    cudaEventElapsedTime(&time_step31, start, stop);
    long long bytes_step31 = vocab_size * sizeof(float) * 2; // read s_g_probs, internal ops
    long long flops_step31 = vocab_size * 4; // prefix sum operations
    timings.push_back({"STEP 3.1. TOP-P*", time_step31, bytes_step31, (long long)(nucleus_bytes), flops_step31});

    // ASSUME that nucleus is no larger than 1024 so we don't have to go back on-host or do multi-passes
    constexpr int final_threads = 1024;
    // assert((nucleus_size <= final_threads) && "Nucleus is too large - implementation for nucleus size up to 1024"); 

    // // // STEP 3.2.: normalize(nucleus) = n_nucleus
    cudaEventRecord(start);

    int final_blocks = (nucleus_size + final_threads - 1) / final_threads;
    float* n_nucleus = nullptr;
    cudaMalloc(&n_nucleus, nucleus_bytes);
    normalize<elemsPerThread, final_threads><<<final_blocks, final_threads>>>(nucleus, n_nucleus, nucleus_size, global_probs_sum);
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float time_step32 = 0.0f;
    cudaEventElapsedTime(&time_step32, start, stop);
    long long bytes_step32 = nucleus_bytes * 2; // read nucleus, write n_nucleus
    long long flops_step32 = nucleus_size * 3; // divide operations
    timings.push_back({"STEP 3.2. NORM", time_step32, bytes_step32, (long long)(nucleus_bytes), flops_step32});



    // // // STEP 3.3.: prefix_sum(n_nucleus) = p_n_nucleus
    cudaEventRecord(start);
    //prefix sum on the normalized probabilities is required as the final random sampling requires a prefix sum
    float* p_n_nucleus = nullptr;
    cudaMalloc(&p_n_nucleus, nucleus_bytes);
    float* dummy = nullptr;
    cudaMalloc(&dummy, sizeof(float));
    prefix_sum<threads><<<1, final_threads>>>(n_nucleus, p_n_nucleus, dummy, nucleus_size);
    cudaDeviceSynchronize();
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float time_step33 = 0.0f;
    cudaEventElapsedTime(&time_step33, start, stop);
    long long bytes_step33 = nucleus_bytes * 2; // read n_nucleus, write p_n_nucleus
    long long flops_step33 = nucleus_size * 2; // prefix sum additions
    timings.push_back({"STEP 3.3. PREFIX SUM", time_step33, bytes_step33, (long long)(nucleus_bytes), flops_step33});

    // // // STEP 4: sample(p_n_nucleus) = sampled_token
    cudaEventRecord(start);
    int* nucleus_sampled_token = nullptr;
    cudaMalloc(&nucleus_sampled_token, sizeof(int));
    warp_sample<<<1, final_threads>>>(p_n_nucleus, seed, nucleus_sampled_token, nucleus_size);
    cudaDeviceSynchronize();
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float time_step4 = 0.0f;
    cudaEventElapsedTime(&time_step4, start, stop);
    long long bytes_step4 = nucleus_bytes; // read p_n_nucleus
    long long flops_step4 = nucleus_size * 3; // LCG, comparisons
    timings.push_back({"STEP 4: Sample", time_step4, bytes_step4, (long long)(sizeof(int)), flops_step4});

    // Copy sampled nucleus index to host, map back to original token via d_indices_in
    int sampled_nucleus_idx = 0;
    cudaMemcpy(&sampled_nucleus_idx, nucleus_sampled_token, sizeof(int), cudaMemcpyDeviceToHost);
    if(sampled_nucleus_idx < nucleus_size) {
        // If caller provided pointer for sampled_prob, copy it to device
        if(sampled_prob != nullptr) {
            std::vector<float> h_p_n_nucleus(nucleus_size);
            cudaMemcpy(h_p_n_nucleus.data(), p_n_nucleus, nucleus_size * sizeof(float), cudaMemcpyDeviceToHost);
            float sampled_prob_val = h_p_n_nucleus[sampled_nucleus_idx];
            cudaMemcpy(sampled_prob, &sampled_prob_val, sizeof(float), cudaMemcpyHostToDevice);
        }
    }
    unsigned int original_token = 0;
    cudaMemcpy(&original_token, d_indices_in + sampled_nucleus_idx, sizeof(unsigned int), cudaMemcpyDeviceToHost);
    
    cudaMemcpy(sampled_token, &original_token, sizeof(unsigned int), cudaMemcpyHostToDevice);

    cudaFree(nucleus_sampled_token);

    // Cleanup
    cudaFree(nucleus);
    cudaFree(n_nucleus);
    cudaFree(p_n_nucleus);
    cudaFree(dummy);
    cudaFree(d_per_block_sums);
    cudaFree(s_g_probs);
    cudaFree(uint32_g_probs);
    cudaFree(d_block_sums);
    cudaFree(d_logits_temp);
    cudaFree(d_p_temp);
    cudaFree(d_seed_temp);
    
    // Print timing results
    printf("\n========== TIMING RESULTS ==========");
    printf("\n%-20s | Time (ms) | Read (MB) | Read (GB/s) | Write (MB) | Write (GB/s) | GFLOPS | AI\n", "Step");
    printf("%-20s-+-----------+-----------+-------------+------------+--------------+--------+-------\n", "");
    
    // H2D transfer
    float h2d_mb = h2d_total_bytes / (1024.0f * 1024);
    float h2d_bw = (h2d_total_bytes / (1024.0f * 1024 * 1024)) / (h2d_total_time / 1000.0f);
    printf("%-20s | %9.3f | %9.3f | %11.1f | %10.3f | %12.1f | %6.1f | %7s\n",
           "H2D Input", h2d_total_time, h2d_mb, h2d_bw, 0.0f, 0.0f, 0.0f, "N/A");
    
    for(const auto& result : timings) {
        result.print();
    }
    float total_time = 0.0f;
    for(const auto& result : timings) total_time += result.time_ms;
    printf("%-20s | %8.3f ms\n", "TOTAL", total_time + h2d_total_time);
    printf("==================================================\n");
    printf("*STEP 3 relates to Nucleus selection, normalization, prefix sum (for the final sample)\n\n");
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

}
