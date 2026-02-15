#include "softmax.h"
#include <cuda_runtime.h>
#include <cfloat>

// Local constants (match top_p.cu)
constexpr int WarpsInBlock = 32;
constexpr int threads = 32 * WarpsInBlock;
constexpr int elemsPerThread = 16;

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

// Kernel 2: exp(x_i - xmax) + partial sum
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

extern "C" void solve_softmax(const float* logits, float* g_probs, int vocab_size, int num_batches){
    
    float* bufA = nullptr;
    float* bufB = nullptr;
    cudaMalloc(&bufA, ((vocab_size + elemsPerThread * threads - 1) / (elemsPerThread * threads)) * num_batches * sizeof(float));
    cudaMalloc(&bufB, ((vocab_size + elemsPerThread * threads - 1) / (elemsPerThread * threads)) * num_batches * sizeof(float));

    const int tile_size = elemsPerThread * threads;
    const int first_blocks = (vocab_size + tile_size - 1) / tile_size;
    dim3 first_grid(first_blocks, 1, num_batches);

    // Kernel 1: reduce max
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
    float *d_xmax = src;

    cudaDeviceSynchronize();

    // Kernel 2: exp(x_i - xmax) + partial sum
    float* exp_output = nullptr;
    float* partial_sum = nullptr;
    cudaMalloc(&exp_output, vocab_size * num_batches * sizeof(float));
    cudaMalloc(&partial_sum, first_blocks * num_batches * sizeof(float));
    exp_and_partial_sum<elemsPerThread, threads><<<first_grid, threads>>>(logits, exp_output, partial_sum, vocab_size, d_xmax);

    cudaDeviceSynchronize();

    // Kernel 3: reduce sum
    float* src_sum = partial_sum;
    float* dst_sum = bufA;
    curr_size = first_blocks;

    while(curr_size > 1){
        int curr_blocks = (curr_size + tile_size - 1) / tile_size;
        dim3 curr_grid(curr_blocks, 1, num_batches);
        reduce_sum_step<elemsPerThread, threads><<<curr_grid, threads>>>(src_sum, dst_sum, curr_size);
        curr_size = curr_blocks;
        float* tmp = dst_sum; dst_sum = src_sum; src_sum = tmp;
    }
    float* d_exp_sum = src_sum;

    // Kernel 4: normalize
    normalize<elemsPerThread, threads><<<first_grid, threads>>>(exp_output, g_probs, vocab_size, d_exp_sum);

    // Cleanup
    cudaFree(bufA);
    cudaFree(bufB);
    cudaFree(exp_output);
    cudaFree(partial_sum);

    cudaDeviceSynchronize();
}

extern "C" void solve_normalize(const float* input, float* output, int N, const float* sums, int vocab_size, int num_batches){
    const int blocks = (vocab_size + threads - 1) / threads;   
    dim3 norm_grid(blocks, 1, num_batches);
    normalize<elemsPerThread, threads><<<norm_grid, threads>>>(input, output, N, sums);
    cudaDeviceSynchronize();
}
