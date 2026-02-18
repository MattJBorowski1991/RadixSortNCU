#include <cuda_runtime.h>
#include "../../../utils/cuda_utils.h"
#include <vector>
#include <assert.h>

constexpr int WarpsInBlock = 32;
constexpr int threads = (32 * WarpsInBlock);

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

extern "C" void solve_radix(
    unsigned int *g_probs, 
    unsigned int *asc_g_probs, unsigned int* asc_indices, 
    int vocab_size, int num_batches){

    //assert(vocab_size % 32 == 0);

    int blocksx = (vocab_size + threads - 1) / threads;
    dim3 grid(blocksx, 1, num_batches);


    unsigned int *d_in, *d_out, *d_indices_in, *d_indices_out;
    CHECK_CUDA(cudaMalloc(&d_in, vocab_size * num_batches * sizeof(unsigned int)));
    CHECK_CUDA(cudaMalloc(&d_out, vocab_size * num_batches * sizeof(unsigned int)));
    CHECK_CUDA(cudaMalloc(&d_indices_in, vocab_size * num_batches * sizeof(unsigned int)));
    CHECK_CUDA(cudaMalloc(&d_indices_out, vocab_size * num_batches * sizeof(unsigned int)));

    CHECK_CUDA(cudaMemcpy(d_in, g_probs, vocab_size * num_batches * sizeof(unsigned int), cudaMemcpyDeviceToDevice));
    CHECK_CUDA(init_indices<threads><<<grid, threads>>>(d_indices_in, vocab_size));
    CHECK_CUDA(init_indices<threads><<<grid, threads>>>(d_indices_out, vocab_size));

    std::vector<unsigned int> h_block_sums(blocksx);
    std::vector<unsigned int> h_offsets(blocksx);

    unsigned int *d_block_sums;
    CHECK_CUDA(cudaMalloc(&d_block_sums, blocksx * num_batches * sizeof(unsigned int)));

    // Store total_ones per batch for each bit iteration
    unsigned int *d_total_ones;
    CHECK_CUDA(cudaMalloc(&d_total_ones, 32 * num_batches * sizeof(unsigned int)));

    for (int iter = 0; iter < 32; iter++) {

        // Compute prefix sums for all batches at once
        CHECK_CUDA(prefix_per_block_desc<threads><<<grid, threads>>>(d_in, d_block_sums, iter, vocab_size));

        // Compute total_ones per batch on host (still necessary for correctness)
        std::vector<unsigned int> h_total_ones(num_batches, 0);
        for(int b = 0; b < num_batches; ++b){
            CHECK_CUDA(cudaMemcpy(h_block_sums.data(), d_block_sums + b * blocksx, blocksx * sizeof(unsigned int), cudaMemcpyDeviceToHost));
            
            unsigned int total_ones = 0;
            for (int i = 0; i < blocksx; i++) {
                h_offsets[i] = total_ones;
                total_ones += h_block_sums[i];
            }
            h_total_ones[b] = total_ones;
            CHECK_CUDA(cudaMemcpy(d_block_sums + b * blocksx, h_offsets.data(), blocksx * sizeof(unsigned int), cudaMemcpyHostToDevice));
        }
        
        // Copy total_ones to device and launch sort for all batches
        CHECK_CUDA(cudaMemcpy(d_total_ones + iter * num_batches, h_total_ones.data(), num_batches * sizeof(unsigned int), cudaMemcpyHostToDevice));
        CHECK_CUDA(radix_sort_asc_kernel<threads><<<grid, threads>>>(d_in, d_out, d_indices_in, d_indices_out, d_block_sums, d_total_ones + iter * num_batches, iter, vocab_size));
        std::swap(d_in, d_out);
        std::swap(d_indices_in, d_indices_out);
    }
  
    CHECK_CUDA(cudaMemcpy(asc_g_probs, d_in, vocab_size * num_batches * sizeof(unsigned int), cudaMemcpyDeviceToDevice));
    CHECK_CUDA(cudaMemcpy(asc_indices, d_indices_in, vocab_size * num_batches * sizeof(unsigned int), cudaMemcpyDeviceToDevice));

    CHECK_CUDA(cudaFree(d_total_ones));
    CHECK_CUDA(cudaFree(d_in));
    CHECK_CUDA(cudaFree(d_out));
    CHECK_CUDA(cudaFree(d_indices_out));
    CHECK_CUDA(cudaFree(d_block_sums));
}