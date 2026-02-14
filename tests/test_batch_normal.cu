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

// Host-side LCG RNG (same as GPU code)
__host__ __device__ __forceinline__ float lcg_uniform(unsigned int seed) {
    unsigned int state = seed;
    state = (1664525u * state + 1013904223u);
    return (float)state / 4294967296.0f;
}

// Compute expected result on host for validation
struct HostResult {
    int nucleus_size;
    int expected_token;
    float sampled_prob;
    std::vector<float> nucleus_values;
    std::vector<float> nucleus_normalized;
    std::vector<float> prefix_sum_values;
    std::vector<float> sorted_probs;
};

HostResult compute_expected_result(const std::vector<float>& logits, float p, int seed, int vocab_size) {
    // Step 1: Softmax
    float max_logit = *std::max_element(logits.begin(), logits.end());
    std::vector<float> exp_logits(vocab_size);
    float sum_exp = 0.0f;
    for(int i = 0; i < vocab_size; i++) {
        exp_logits[i] = std::exp(logits[i] - max_logit);
        sum_exp += exp_logits[i];
    }
    std::vector<float> probs(vocab_size);
    for(int i = 0; i < vocab_size; i++) {
        probs[i] = exp_logits[i] / sum_exp;
    }
    
    // Step 2: Sort probabilities descending with index tracking
    std::vector<std::pair<float, int>> prob_idx;
    for(int i = 0; i < vocab_size; i++) {
        prob_idx.push_back({probs[i], i});
    }
    std::sort(prob_idx.begin(), prob_idx.end(), [](const auto& a, const auto& b) {
        return a.first > b.first;
    });
    
    // Step 3: Find nucleus size
    float cumsum = 0.0f;
    int nucleus_size = 0;
    for(int i = 0; i < vocab_size; i++) {
        cumsum += prob_idx[i].first;
        nucleus_size = i + 1;
        if(cumsum >= p) break;
    }
    nucleus_size = std::min(nucleus_size, 1000);  // Cap nucleus to top-1000
    
    // Step 4: Compute normalized nucleus probabilities
    float nucleus_sum = 0.0f;
    for(int i = 0; i < nucleus_size; i++) {
        nucleus_sum += prob_idx[i].first;
    }
    std::vector<float> nucleus_probs(nucleus_size);
    for(int i = 0; i < nucleus_size; i++) {
        nucleus_probs[i] = prob_idx[i].first / nucleus_sum;
    }
    
    // Step 5: Prefix sum
    std::vector<float> prefix_sum(nucleus_size);
    prefix_sum[0] = nucleus_probs[0];
    for(int i = 1; i < nucleus_size; i++) {
        prefix_sum[i] = prefix_sum[i-1] + nucleus_probs[i];
    }
    
    // Step 6: Generate sampled probability using same LCG
    float sampled_prob = lcg_uniform(seed);
    
    // Step 7: Find token
    int sampled_nucleus_idx = 0;
    for(int i = 0; i < nucleus_size; i++) {
        if(prefix_sum[i] >= sampled_prob) {
            sampled_nucleus_idx = i;
            break;
        }
    }
    
    // Step 8: Map back to original token
    int expected_token = prob_idx[sampled_nucleus_idx].second;
    
    // Store for debug output
    HostResult result;
    result.nucleus_size = nucleus_size;
    result.expected_token = expected_token;
    result.sampled_prob = sampled_prob;
    result.nucleus_values = std::vector<float>(nucleus_probs.begin(), nucleus_probs.begin() + std::min(10, nucleus_size));
    result.nucleus_normalized = std::vector<float>(nucleus_probs.begin(), nucleus_probs.begin() + std::min(10, nucleus_size));
    result.prefix_sum_values = std::vector<float>(prefix_sum.begin(), prefix_sum.begin() + std::min(10, nucleus_size));
    result.sorted_probs = std::vector<float>(nucleus_probs.begin(), nucleus_probs.begin() + std::min(10, nucleus_size));
    
    return result;
}

void run_batch_normal_test(int num_batches, int vocab_size, float variance, float p_val, int rng_seed) {
    
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
    solve(d_logits, d_p, d_seed, d_sampled_token, vocab_size, num_batches);
    
    // Copy results back
    std::vector<int> h_sampled_tokens(num_batches);
    cudaMemcpy(h_sampled_tokens.data(), d_sampled_token, num_batches * sizeof(int), cudaMemcpyDeviceToHost);
    
    // Compute expected results on host for each batch
    std::vector<HostResult> expected_results(num_batches);
    for(int b = 0; b < num_batches; ++b) {
        std::vector<float> batch_logits(h_logits.begin() + b * vocab_size, 
                                        h_logits.begin() + (b + 1) * vocab_size);
        expected_results[b] = compute_expected_result(batch_logits, h_p[b], h_seed[b], vocab_size);
    }
    
    // Print results with validation
    std::cout << "\nResults with Host-Side Validation:" << std::endl;
    std::cout << std::string(140, '-') << std::endl;
    std::cout << "Batch | GPU Token | Expected | Match | Nucleus | Sampled Prob | Host Seed | GPU Seed | Status" << std::endl;
    std::cout << std::string(140, '-') << std::endl;
    
    int valid_count = 0;
    int correct_count = 0;
    int in_bounds_count = 0;
    for(int b = 0; b < num_batches; ++b) {
        int gpu_token = h_sampled_tokens[b];
        int expected_token = expected_results[b].expected_token;
        int nucleus_size = expected_results[b].nucleus_size;
        float sampled_prob = expected_results[b].sampled_prob;
        
        bool in_bounds = (gpu_token >= 0 && gpu_token < vocab_size);
        bool matches = (gpu_token == expected_token);
        
        std::cout << std::setw(5) << b << " | ";
        std::cout << std::setw(9) << gpu_token << " | ";
        std::cout << std::setw(8) << expected_token << " | ";
        
        if(matches) {
            std::cout << "✓ YES | ";
            correct_count++;
        } else {
            std::cout << "✗ NO  | ";
        }
        
        std::cout << std::setw(7) << nucleus_size << " | ";
        std::cout << std::fixed << std::setprecision(6) << std::setw(12) << sampled_prob << " | ";
        std::cout << std::setw(9) << h_seed[b] << " | ";
        std::cout << std::setw(8) << (h_seed[b] >> 16) << " | ";
        
        if(!in_bounds) {
            std::cout << "FAIL (out of bounds)";
        } else {
            in_bounds_count++;
            if(matches) {
                std::cout << "PASS";
                valid_count++;
            } else {
                std::cout << "FAIL (mismatch)";
            }
        }
        
        std::cout << std::endl;
    }
    std::cout << std::string(140, '-') << std::endl;
    
    // Summary
    std::cout << "\n✓ SUMMARY:" << std::endl;
    std::cout << "  - Tokens in bounds: " << in_bounds_count << "/" << num_batches << std::endl;
    std::cout << "  - Correct matches: " << correct_count << "/" << num_batches << std::endl;
    
    if(correct_count == num_batches) {
        std::cout << "  - Overall: PASS ✓" << std::endl;
    } else {
        std::cout << "  - Overall: FAIL ✗" << std::endl;
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
            std::cout << "  --vocab_size, -v <N>   Vocabulary size (> 0)" << std::endl;
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
    if(vocab_size <= 0) {
        std::cerr << "Error: vocab_size must be > 0" << std::endl;
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
