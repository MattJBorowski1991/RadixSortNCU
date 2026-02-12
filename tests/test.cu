#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cstdlib>
#include <algorithm>
#include <cmath>
#include <ctime>

// nvcc tests/test.cu top_p_sampling.cu -o test 2>&1 && ./test normal 100 0.95

extern "C" void solve(
    const float* logits,
    const float* p,
    const int* seed,
    int* sampled_token,
    int vocab_size,
    float* sampled_prob = nullptr
);

void run_test(int vocab_size, float p_val) {
    std::vector<float> h_logits(vocab_size);
    for (int i = 0; i < vocab_size; ++i) {
        h_logits[i] = static_cast<float>(i);
    }
    std::vector<float> h_p = {p_val};
    std::vector<int> h_seed = {123};

    // DEBUG: Print input logits
    std::cout << "\n========== DEBUG: On-host calc values ==========" << std::endl;
    std::cout << "1. Input logits: ";
    for(float l : h_logits) std::cout << l << " ";
    std::cout << std::endl;

    // DEBUG: Calculate and print softmax probabilities
    float max_logit = *std::max_element(h_logits.begin(), h_logits.end());
    std::vector<float> exp_logits(vocab_size);
    float sum_exp = 0.0f;
    for(int i = 0; i < vocab_size; i++) {
        exp_logits[i] = std::exp(h_logits[i] - max_logit);
        sum_exp += exp_logits[i];
    }
    std::vector<float> probs(vocab_size);
    for(int i = 0; i < vocab_size; i++) {
        probs[i] = exp_logits[i] / sum_exp;
    }
    std::cout << "2. Probabilities (softmax): ";
    for(float p : probs) std::cout << p << " ";
    std::cout << std::endl;

    // DEBUG: Sort probabilities descending and get sorted probs
    std::vector<std::pair<float, int>> prob_idx;
    for(int i = 0; i < vocab_size; i++) {
        prob_idx.push_back({probs[i], i});
    }
    std::sort(prob_idx.begin(), prob_idx.end(), [](const auto& a, const auto& b) {
        return a.first > b.first;
    });
    std::vector<float> sorted_probs(vocab_size);
    std::vector<int> sorted_indices(vocab_size);
    for(int i = 0; i < vocab_size; i++) {
        sorted_probs[i] = prob_idx[i].first;
        sorted_indices[i] = prob_idx[i].second;
    }
    std::cout << "3. Sorted probs (descending): ";
    for(float p : sorted_probs) std::cout << p << " ";
    std::cout << std::endl;

    // DEBUG: Calculate and print prefix sum
    std::vector<float> prefix_sum_probs(vocab_size);
    float cumsum = 0.0f;
    for(int i = 0; i < vocab_size; i++) {
        cumsum += sorted_probs[i];
        prefix_sum_probs[i] = cumsum;
    }
    std::cout << "4. Prefix sum of sorted probs: ";
    for(float p : prefix_sum_probs) std::cout << p << " ";
    std::cout << std::endl;

    // DEBUG: Calculate sampled_prob from seed
    int seed = h_seed[0];
    unsigned int state = (1664525u * seed + 1013904223u);
    float sampled_prob = (float)state / 4294967296.0f;
    std::cout << "5. Sampled prob (seed=" << seed << "): " << sampled_prob << std::endl;

    // DEBUG: Find selected token
    int selected_sorted_pos = 0;
    for(int i = 0; i < vocab_size; i++) {
        if(prefix_sum_probs[i] >= sampled_prob) {
            selected_sorted_pos = i;
            break;
        }
    }
    int selected_token = sorted_indices[selected_sorted_pos];
    std::cout << "6. Selected token (original index): " << selected_token << std::endl;
    std::cout << "   (at sorted position " << selected_sorted_pos << ")" << std::endl;

    // Allocate device memory
    float* d_logits;
    float* d_p;
    int* d_seed;
    int* d_sampled_token;
    float* d_sampled_prob;

    cudaMalloc(&d_logits, vocab_size * sizeof(float));
    cudaMalloc(&d_p, sizeof(float));
    cudaMalloc(&d_seed, sizeof(int));
    cudaMalloc(&d_sampled_token, sizeof(int));
    cudaMalloc(&d_sampled_prob, sizeof(float));

    // Copy to device
    cudaMemcpy(d_logits, h_logits.data(), vocab_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_p, h_p.data(), sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_seed, h_seed.data(), sizeof(int), cudaMemcpyHostToDevice);

    // Call solve
    solve(d_logits, d_p, d_seed, d_sampled_token, vocab_size, d_sampled_prob);

    // Copy result back
    int h_sampled_token;
    float h_sampled_prob;
    cudaMemcpy(&h_sampled_token, d_sampled_token, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_sampled_prob, d_sampled_prob, sizeof(float), cudaMemcpyDeviceToHost);

    // Print result
    std::cout << "Result from solve(): Sampled token: " << h_sampled_token << std::endl;
    std::cout << "Sampled probability (from curand): " << h_sampled_prob << std::endl;
    if(h_sampled_token == selected_token) {
        std::cout << "✓ MATCHES expected token!" << std::endl;
    } else {
        std::cout << "✗ MISMATCH! Expected: " << selected_token << ", Got: " << h_sampled_token << std::endl;
    }

    // Cleanup
    cudaFree(d_logits);
    cudaFree(d_p);
    cudaFree(d_seed);
    cudaFree(d_sampled_token);
    cudaFree(d_sampled_prob);
}

void run_original_test() {
    // Original test inputs
    std::vector<float> h_logits = {1.0f, 2.0f, 3.0f};
    std::vector<float> h_p = {0.949999988079071f};
    std::vector<int> h_seed = {123};
    int vocab_size = 3;

    // Print input array
    std::cout << "\n========== Input Array (run_original_test) ==========" << std::endl;
    std::cout << "Input logits: ";
    for(float l : h_logits) std::cout << l << " ";
    std::cout << std::endl;

    // Allocate device memory
    float* d_logits;
    float* d_p;
    int* d_seed;
    int* d_sampled_token;

    cudaMalloc(&d_logits, vocab_size * sizeof(float));
    cudaMalloc(&d_p, sizeof(float));
    cudaMalloc(&d_seed, sizeof(int));
    cudaMalloc(&d_sampled_token, sizeof(int));

    // Copy to device
    cudaMemcpy(d_logits, h_logits.data(), vocab_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_p, h_p.data(), sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_seed, h_seed.data(), sizeof(int), cudaMemcpyHostToDevice);

    // Call solve
    solve(d_logits, d_p, d_seed, d_sampled_token, vocab_size);

    // Copy result back
    int h_sampled_token;
    cudaMemcpy(&h_sampled_token, d_sampled_token, sizeof(int), cudaMemcpyDeviceToHost);

    // Print result
    std::cout << "Sampled token: " << h_sampled_token << std::endl;
    std::cout << "Expected: 1" << std::endl;
    if (h_sampled_token == 1) {
        std::cout << "Test PASSED" << std::endl;
    } else {
        std::cout << "Test FAILED" << std::endl;
    }

    // Cleanup
    cudaFree(d_logits);
    cudaFree(d_p);
    cudaFree(d_seed);
    cudaFree(d_sampled_token);
}

void run_failing_test() {
    // Failing test case from platform
    std::vector<float> h_logits = {
        0.4966076910495758f,
        2.36181378364563f,
        1.8229269981384277f,
        -0.9366657733917236f,
        0.13903221487998962f,
        0.010955136269330978f,
        -0.48935171961784363f,
        -0.6667436957359314f,
        0.19667406380176544f,
        0.18078237771987915f
    };
    std::vector<float> h_p = {0.8999999761581421f};
    std::vector<int> h_seed = {456};
    int vocab_size = 10;

    // Print input array
    std::cout << "\n========== Input Array (run_failing_test) ==========" << std::endl;
    std::cout << "Input logits: ";
    for(float l : h_logits) std::cout << l << " ";
    std::cout << std::endl;

    // Allocate device memory
    float* d_logits;
    float* d_p;
    int* d_seed;
    int* d_sampled_token;

    cudaMalloc(&d_logits, vocab_size * sizeof(float));
    cudaMalloc(&d_p, sizeof(float));
    cudaMalloc(&d_seed, sizeof(int));
    cudaMalloc(&d_sampled_token, sizeof(int));

    // Copy to device
    cudaMemcpy(d_logits, h_logits.data(), vocab_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_p, h_p.data(), sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_seed, h_seed.data(), sizeof(int), cudaMemcpyHostToDevice);

    // Call solve
    solve(d_logits, d_p, d_seed, d_sampled_token, vocab_size);

    // Copy result back
    int h_sampled_token;
    cudaMemcpy(&h_sampled_token, d_sampled_token, sizeof(int), cudaMemcpyDeviceToHost);

    // Print result
    std::cout << "Sampled token: " << h_sampled_token << std::endl;
    std::cout << "Expected: 1" << std::endl;
    if (h_sampled_token == 1) {
        std::cout << "Test PASSED" << std::endl;
    } else {
        std::cout << "Test FAILED" << std::endl;
    }

    // Cleanup
    cudaFree(d_logits);
    cudaFree(d_p);
    cudaFree(d_seed);
    cudaFree(d_sampled_token);
}

// Box-Muller transform to generate normally distributed random numbers
float box_muller(float u1, float u2) {
    return std::sqrt(-2.0f * std::log(u1)) * std::cos(2.0f * M_PI * u2);
}

void run_normal_test(int vocab_size, float p_val, int seed_val = -1) {
    // If seed_val is -1, use current time for random seed
    if(seed_val == -1) {
        seed_val = static_cast<int>(time(nullptr) % 2147483647);
    }
    
    // Generate N(0, 10) distributed logits
    std::vector<float> h_logits(vocab_size);
    
    // Simple LCG for random number generation
    unsigned int state = seed_val;
    auto lcg = [&state]() -> float {
        state = (1664525u * state + 1013904223u);
        return (float)state / 4294967296.0f;
    };
    
    for (int i = 0; i < vocab_size; ++i) {
        float u1 = lcg();
        float u2 = lcg();
        u1 = std::max(u1, 1e-7f); // Avoid log(0)
        h_logits[i] = box_muller(u1, u2) * 10.0f; // N(0, 10)
    }
    
    std::vector<float> h_p = {p_val};
    std::vector<int> h_seed = {123};

    // DEBUG: Print input logits
    std::cout << "\n========== DEBUG: On-host calculated values (N(0,10) test, seed=" << seed_val << ") ==========" << std::endl;
    std::cout << "1. Input logits (N(0,10)): ";
    for(int i = 0; i < std::min(10, vocab_size); i++) std::cout << h_logits[i] << " ";
    if(vocab_size > 10) std::cout << "... [" << vocab_size << " total]";
    std::cout << std::endl;

    // Calculate softmax probabilities
    float max_logit = *std::max_element(h_logits.begin(), h_logits.end());
    std::vector<float> exp_logits(vocab_size);
    float sum_exp = 0.0f;
    for(int i = 0; i < vocab_size; i++) {
        exp_logits[i] = std::exp(h_logits[i] - max_logit);
        sum_exp += exp_logits[i];
    }
    std::vector<float> probs(vocab_size);
    for(int i = 0; i < vocab_size; i++) {
        probs[i] = exp_logits[i] / sum_exp;
    }
    std::cout << "2. Probabilities (softmax): ";
    for(int i = 0; i < std::min(10, vocab_size); i++) std::cout << probs[i] << " ";
    if(vocab_size > 10) std::cout << "... [" << vocab_size << " total]";
    std::cout << std::endl;

    // Sort probabilities descending and get sorted probs
    std::vector<std::pair<float, int>> prob_idx;
    for(int i = 0; i < vocab_size; i++) {
        prob_idx.push_back({probs[i], i});
    }
    std::sort(prob_idx.begin(), prob_idx.end(), [](const auto& a, const auto& b) {
        return a.first > b.first;
    });
    std::vector<float> sorted_probs(vocab_size);
    std::vector<int> sorted_indices(vocab_size);
    for(int i = 0; i < vocab_size; i++) {
        sorted_probs[i] = prob_idx[i].first;
        sorted_indices[i] = prob_idx[i].second;
    }
    std::cout << "3. Sorted probs (descending): ";
    for(int i = 0; i < std::min(10, vocab_size); i++) std::cout << sorted_probs[i] << " ";
    if(vocab_size > 10) std::cout << "... [" << vocab_size << " total]";
    std::cout << std::endl;

    // Calculate prefix sum
    std::vector<float> prefix_sum_probs(vocab_size);
    float cumsum = 0.0f;
    for(int i = 0; i < vocab_size; i++) {
        cumsum += sorted_probs[i];
        prefix_sum_probs[i] = cumsum;
    }
    std::cout << "4. Prefix sum of sorted probs: ";
    for(int i = 0; i < std::min(10, vocab_size); i++) std::cout << prefix_sum_probs[i] << " ";
    if(vocab_size > 10) std::cout << "... [" << vocab_size << " total]";
    std::cout << std::endl;

    // Calculate sampled_prob from seed
    int seed = h_seed[0];
    unsigned int rng_state = (1664525u * seed + 1013904223u);
    float sampled_prob = (float)rng_state / 4294967296.0f;
    std::cout << "5. Sampled prob (seed=" << seed << "): " << sampled_prob << std::endl;

    // Find selected token
    int selected_sorted_pos = 0;
    for(int i = 0; i < vocab_size; i++) {
        if(prefix_sum_probs[i] >= sampled_prob) {
            selected_sorted_pos = i;
            break;
        }
    }
    int selected_token = sorted_indices[selected_sorted_pos];
    std::cout << "6. Selected token (original index): " << selected_token << std::endl;
    std::cout << "   (at sorted position " << selected_sorted_pos << ")" << std::endl;

    // Allocate device memory
    float* d_logits;
    float* d_p;
    int* d_seed;
    int* d_sampled_token;
    float* d_sampled_prob;

    cudaMalloc(&d_logits, vocab_size * sizeof(float));
    cudaMalloc(&d_p, sizeof(float));
    cudaMalloc(&d_seed, sizeof(int));
    cudaMalloc(&d_sampled_token, sizeof(int));
    cudaMalloc(&d_sampled_prob, sizeof(float));

    // Copy to device
    cudaMemcpy(d_logits, h_logits.data(), vocab_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_p, h_p.data(), sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_seed, h_seed.data(), sizeof(int), cudaMemcpyHostToDevice);

    // Call solve
    solve(d_logits, d_p, d_seed, d_sampled_token, vocab_size, d_sampled_prob);

    // Copy result back
    int h_sampled_token;
    float h_sampled_prob;
    cudaMemcpy(&h_sampled_token, d_sampled_token, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_sampled_prob, d_sampled_prob, sizeof(float), cudaMemcpyDeviceToHost);

    // Print result
    std::cout << "Result from solve(): Sampled token: " << h_sampled_token << std::endl;
    std::cout << "Sampled probability (from curand): " << h_sampled_prob << std::endl;
    if(h_sampled_token == selected_token) {
        std::cout << "✓ MATCHES expected token!" << std::endl;
    } else {
        std::cout << "✗ MISMATCH! Expected: " << selected_token << ", Got: " << h_sampled_token << std::endl;
    }

    // Cleanup
    cudaFree(d_logits);
    cudaFree(d_p);
    cudaFree(d_seed);
    cudaFree(d_sampled_token);
    cudaFree(d_sampled_prob);
}

int main(int argc, char* argv[]) {
    if (argc == 2 && std::string(argv[1]) == "original") {
        // Run original test: ./test original
        std::cout << "Running original test (vocab_size=3):" << std::endl;
        run_original_test();
        return 0;
    } else if (argc == 2 && std::string(argv[1]) == "failing") {
        // Run failing test: ./test failing
        std::cout << "Running failing test (vocab_size=10, seed=456):" << std::endl;
        run_failing_test();
        return 0;
    } else if (argc >= 3 && std::string(argv[1]) == "normal") {
        // Run normal distribution test: ./test normal <vocab_size> [p] [--seed <seed>]
        int vocab_size = atoi(argv[2]);
        float p_val = 0.95f;
        int seed_val = -1;
        
        // Parse optional p and seed arguments
        for(int i = 3; i < argc; i++) {
            if(std::string(argv[i]) == "--seed" && i + 1 < argc) {
                seed_val = atoi(argv[i+1]);
                i++; // Skip next argument as we've consumed it
            } else if(i == 3) {
                // Third argument without --seed flag is assumed to be p value
                p_val = atof(argv[3]);
            }
        }
        
        std::cout << "Running N(0,10) test: vocab_size=" << vocab_size << ", p=" << p_val;
        if(seed_val != -1) std::cout << ", seed=" << seed_val;
        std::cout << std::endl;
        run_normal_test(vocab_size, p_val, seed_val);
        return 0;
    } else if (argc == 3) {
        // Run specific test: ./test <vocab_size> <p>
        int vocab_size = atoi(argv[1]);
        float p_val = atof(argv[2]);
        std::cout << "Running specific test: vocab_size=" << vocab_size << ", p=" << p_val << std::endl;
        run_test(vocab_size, p_val);
        return 0;
    }

    // Default: run all tests
    std::cout << "Running all tests:" << std::endl;
    run_original_test();
    std::cout << std::endl;
    run_failing_test();

    // Additional tests
    std::cout << "\nRunning additional tests:" << std::endl;
    std::vector<int> sizes = {10, 100, 1000, 2000};
    std::vector<float> p_vals = {0.92f, 0.95f};
    for (int size : sizes) {
        for (float p : p_vals) {
            run_test(size, p);
        }
    }

    // Normal distribution tests
    std::cout << "\nRunning N(0,10) tests:" << std::endl;
    std::vector<int> normal_sizes = {10, 100, 1000};
    for (int size : normal_sizes) {
        run_normal_test(size, 0.95f);
    }

    return 0;
}