#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cstdlib>
#include <algorithm>
#include <cmath>
#include <random>
#include <iomanip>

// Test with arbitrary batch size and vocab size
// Logits generated from N(0, v) normal distribution
// Usage: ./test_batch_normal <batch_size> <vocab_size> <variance> [p_value] [seed]
// Example: ./test_batch_normal 4 256 1.0 0.95 12345

extern "C" void solve(
    const float* logits,
    const float* p,
    const int* seed,
    int* sampled_token,
    int vocab_size,
    int num_batches
);

void run_batch_normal_test(int num_batches, int vocab_size, float variance, float p_val, int rng_seed) {
    std::cout << "\n========== BATCH TEST (Normal Distribution Logits) ==========" << std::endl;
    std::cout << "Configuration:" << std::endl;
    std::cout << "  - Batch size: " << num_batches << std::endl;
    std::cout << "  - Vocab size: " << vocab_size << std::endl;
    std::cout << "  - Logits distribution: N(0, " << variance << ")" << std::endl;
    std::cout << "  - p value: " << p_val << " (per batch)" << std::endl;
    std::cout << "  - RNG seed: " << rng_seed << std::endl;
    
    // Generate random logits from N(0, variance)
    std::mt19937 gen(rng_seed);
    std::normal_distribution<float> dist(0.0f, std::sqrt(variance));
    
    std::vector<float> h_logits(vocab_size * num_batches);
    for(int i = 0; i < vocab_size * num_batches; ++i) {
        h_logits[i] = dist(gen);
    }
    
    // Create p values for each batch (all same)
    std::vector<float> h_p(num_batches, p_val);
    
    // Create seeds for each batch (different for each)
    std::vector<int> h_seed(num_batches);
    for(int b = 0; b < num_batches; ++b) {
        h_seed[b] = rng_seed + b * 1000;  // Offset seeds for each batch
    }
    
    std::cout << "\nGenerated logits stats:" << std::endl;
    float min_logit = h_logits[0], max_logit = h_logits[0], mean_logit = 0.0f;
    for(auto l : h_logits) {
        min_logit = std::min(min_logit, l);
        max_logit = std::max(max_logit, l);
        mean_logit += l;
    }
    mean_logit /= h_logits.size();
    float var_logits = 0.0f;
    for(auto l : h_logits) {
        var_logits += (l - mean_logit) * (l - mean_logit);
    }
    var_logits /= h_logits.size();
    
    std::cout << "  - Min: " << std::fixed << std::setprecision(4) << min_logit << std::endl;
    std::cout << "  - Max: " << std::fixed << std::setprecision(4) << max_logit << std::endl;
    std::cout << "  - Mean: " << std::fixed << std::setprecision(4) << mean_logit << std::endl;
    std::cout << "  - Variance: " << std::fixed << std::setprecision(4) << var_logits << std::endl;
    
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
    int valid_count = 0;
    for(int b = 0; b < num_batches; ++b) {
        std::cout << "  Batch " << std::setw(3) << b << ": Sampled token = " << std::setw(5) << h_sampled_tokens[b];
        if(h_sampled_tokens[b] >= 0 && h_sampled_tokens[b] < vocab_size) {
            std::cout << " ✓ (valid)";
            valid_count++;
        } else {
            std::cout << " ✗ (INVALID - out of bounds!)";
        }
        std::cout << std::endl;
    }
    
    std::cout << "\n✓ SUMMARY: " << valid_count << "/" << num_batches << " batches produced valid results";
    if(valid_count == num_batches) {
        std::cout << " - PASS" << std::endl;
    } else {
        std::cout << " - FAIL" << std::endl;
    }
    
    // Cleanup
    cudaFree(d_logits);
    cudaFree(d_p);
    cudaFree(d_seed);
    cudaFree(d_sampled_token);
}

int main(int argc, char* argv[]) {
    // Default values
    int batch_size = -1;
    int vocab_size = -1;
    float variance = -1.0f;
    float p_value = 0.95f;
    int seed = 12345;
    
    // Parse command line arguments
    for(int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        
        if(arg == "--batches" || arg == "-b") {
            if(i + 1 < argc) batch_size = atoi(argv[++i]);
        }
        else if(arg == "--vocab_size" || arg == "-v") {
            if(i + 1 < argc) vocab_size = atoi(argv[++i]);
        }
        else if(arg == "--variance" || arg == "--var") {
            if(i + 1 < argc) variance = atof(argv[++i]);
        }
        else if(arg == "--p_value" || arg == "--p") {
            if(i + 1 < argc) p_value = atof(argv[++i]);
        }
        else if(arg == "--seed" || arg == "-s") {
            if(i + 1 < argc) seed = atoi(argv[++i]);
        }
        else if(arg == "--help" || arg == "-h") {
            std::cout << "Usage: " << argv[0] << " [options]" << std::endl;
            std::cout << "\nRequired Options:" << std::endl;
            std::cout << "  --batches, -b <N>       Number of batches (1-10000)" << std::endl;
            std::cout << "  --vocab_size, -v <N>   Vocabulary size (1-1000000)" << std::endl;
            std::cout << "  --variance, --var <V>  Variance of N(0, V) distribution (> 0)" << std::endl;
            std::cout << "\nOptional Options:" << std::endl;
            std::cout << "  --p_value, --p <P>      Nucleus p threshold (default 0.95)" << std::endl;
            std::cout << "  --seed, -s <S>          Random seed (default 12345)" << std::endl;
            std::cout << "\nExamples:" << std::endl;
            std::cout << "  " << argv[0] << " --batches 4 --vocab_size 256 --variance 1.0" << std::endl;
            std::cout << "  " << argv[0] << " -b 2 -v 512 --var 0.5 --p 0.90 -s 42" << std::endl;
            return 0;
        }
    }
    
    // Check required arguments
    if(batch_size == -1) {
        std::cerr << "Error: --batches is required" << std::endl;
        std::cerr << "Use --help for usage information" << std::endl;
        return 1;
    }
    if(vocab_size == -1) {
        std::cerr << "Error: --vocab_size is required" << std::endl;
        std::cerr << "Use --help for usage information" << std::endl;
        return 1;
    }
    if(variance == -1.0f) {
        std::cerr << "Error: --variance is required" << std::endl;
        std::cerr << "Use --help for usage information" << std::endl;
        return 1;
    }
    
    // Validate inputs
    if(batch_size <= 0 || batch_size > 10000) {
        std::cerr << "Error: batch_size must be in range (1, 10000]" << std::endl;
        return 1;
    }
    if(vocab_size <= 0 || vocab_size > 1000000) {
        std::cerr << "Error: vocab_size must be in range (1, 1000000]" << std::endl;
        return 1;
    }
    if(variance <= 0) {
        std::cerr << "Error: variance must be > 0" << std::endl;
        return 1;
    }
    if(p_value <= 0 || p_value > 1) {
        std::cerr << "Error: p_value must be in range (0, 1]" << std::endl;
        return 1;
    }
    
    run_batch_normal_test(batch_size, vocab_size, variance, p_value, seed);
    return 0;
}
