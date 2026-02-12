#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <cstdint>

extern "C" void solve(
    const float* logits,
    const float* p,
    const int* seed,
    int* sampled_token,
    int vocab_size,
    float* sampled_prob = nullptr
);

void run_profile_test(const char* data_file = "profile_logits_100k.bin") {
    // Load pre-generated logits from binary file
    std::ifstream file(data_file, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open profile data file: " << data_file << std::endl;
        std::cerr << "Please run: g++ generate_profile_data.cpp -o gen_data -lm && ./gen_data" << std::endl;
        return;
    }
    
    // Read metadata
    int32_t vocab_size;
    int32_t seed_val;
    file.read(reinterpret_cast<char*>(&vocab_size), sizeof(int32_t));
    file.read(reinterpret_cast<char*>(&seed_val), sizeof(int32_t));
    
    // Read logits
    std::vector<float> h_logits(vocab_size);
    file.read(reinterpret_cast<char*>(h_logits.data()), vocab_size * sizeof(float));
    file.close();
    
    std::cout << "\n========== PROFILING TEST ==========" << std::endl;
    std::cout << "Loaded " << vocab_size << " logits from " << data_file << std::endl;
    
    std::vector<float> h_p = {0.92f};
    std::vector<int> h_seed = {123};
    
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
    
    // Call solve (this will print timing results)
    solve(d_logits, d_p, d_seed, d_sampled_token, vocab_size, d_sampled_prob);
    
    // Copy result back
    int h_sampled_token;
    float h_sampled_prob;
    cudaMemcpy(&h_sampled_token, d_sampled_token, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_sampled_prob, d_sampled_prob, sizeof(float), cudaMemcpyDeviceToHost);
    
    std::cout << "Sampled token: " << h_sampled_token << std::endl;
    std::cout << "Sampled probability: " << h_sampled_prob << std::endl;
    
    // Cleanup
    cudaFree(d_logits);
    cudaFree(d_p);
    cudaFree(d_seed);
    cudaFree(d_sampled_token);
    cudaFree(d_sampled_prob);
}

int main(int argc, char* argv[]) {
    if (argc == 1) {
        // Default: profile with generated data
        run_profile_test();
    } else if (argc == 2) {
        // Custom data file
        run_profile_test(argv[1]);
    } else {
        std::cerr << "Usage: " << argv[0] << " [data_file.bin]" << std::endl;
        return 1;
    }
    return 0;
}
