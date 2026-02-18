#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cstdlib>
#include <algorithm>
#include <cmath>
#include <ctime>

// Test with 2 batches, vocab_size=128
// nvcc tests/test_batch2.cu ../top_p.cu -o test_batch2 && ./test_batch2

extern "C" void solve(
    const float* logits,
    const float* p,
    const int* seed,
    int* sampled_token,
    int vocab_size,
    int num_batches
);

void run_batch_test() {
    const int vocab_size = 128;
    const int num_batches = 2;
    
    std::cout << "\n========== BATCH TEST (2 batches, vocab_size=128) ==========" << std::endl;
    
    // Create logits for both batches
    std::vector<float> h_logits(vocab_size * num_batches);
    for(int b = 0; b < num_batches; ++b) {
        for(int i = 0; i < vocab_size; ++i) {
            // Different logits for each batch to ensure they're processed independently
            h_logits[b * vocab_size + i] = static_cast<float>(i) + (b * 10.0f);
        }
    }
    
    // Create p values for each batch
    std::vector<float> h_p = {0.95f, 0.90f};
    
    // Create seeds for each batch
    std::vector<int> h_seed = {123, 456};
    
    std::cout << "Input Configuration:" << std::endl;
    std::cout << "  - Batch 0: p=" << h_p[0] << ", seed=" << h_seed[0] << std::endl;
    std::cout << "  - Batch 1: p=" << h_p[1] << ", seed=" << h_seed[1] << std::endl;
    std::cout << "  - Vocab size: " << vocab_size << std::endl;
    std::cout << "  - Total batches: " << num_batches << std::endl;
    
    // Allocate device memory
    float* d_logits;
    float* d_p;
    int* d_seed;
    int* d_sampled_token;
    
    cudaMalloc(&d_logits, vocab_size * num_batches * sizeof(float));
    cudaMalloc(&d_p, num_batches * sizeof(float));
    cudaMalloc(&d_seed, num_batches * sizeof(int));
    cudaMalloc(&d_sampled_token, num_batches * sizeof(int));
    
    // Copy to device
    cudaMemcpy(d_logits, h_logits.data(), vocab_size * num_batches * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_p, h_p.data(), num_batches * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_seed, h_seed.data(), num_batches * sizeof(int), cudaMemcpyHostToDevice);
    
    // Call solve with batching support
    std::cout << "\nCalling solve()..." << std::endl;
    solve(d_logits, d_p, d_seed, d_sampled_token, vocab_size, num_batches);
    
    // Copy results back
    std::vector<int> h_sampled_tokens(num_batches);
    cudaMemcpy(h_sampled_tokens.data(), d_sampled_token, num_batches * sizeof(int), cudaMemcpyDeviceToHost);
    
    // Print results
    std::cout << "\nResults:" << std::endl;
    for(int b = 0; b < num_batches; ++b) {
        std::cout << "  Batch " << b << ": Sampled token = " << h_sampled_tokens[b];
        if(h_sampled_tokens[b] >= 0 && h_sampled_tokens[b] < vocab_size) {
            std::cout << " ✓ (valid)" << std::endl;
        } else {
            std::cout << " ✗ (INVALID - out of bounds!)" << std::endl;
        }
    }
    
    // Verify results are different (with high probability they should be due to different seeds/p values)
    if(h_sampled_tokens[0] != h_sampled_tokens[1]) {
        std::cout << "\n✓ PASS: Batches produced different results as expected" << std::endl;
    } else {
        std::cout << "\n⚠ WARNING: Batches produced same result (possible but unlikely)" << std::endl;
    }
    
    // Cleanup
    cudaFree(d_logits);
    cudaFree(d_p);
    cudaFree(d_seed);
    cudaFree(d_sampled_token);
}

int main(int argc, char* argv[]) {
    std::cout << "Testing batched nucleus sampling (2 batches, vocab_size=128)" << std::endl;
    run_batch_test();
    return 0;
}
