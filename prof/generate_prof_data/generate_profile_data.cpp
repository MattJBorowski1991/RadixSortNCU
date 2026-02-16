#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <cstdint>

// Generate N(0, 10) distributed random values using Box-Muller
float box_muller(float u1, float u2) {
    return std::sqrt(-2.0f * std::log(u1)) * std::cos(2.0f * M_PI * u2);
}

int main() {
    const int VOCAB_SIZE = 100000;
    const int SEED = 123;
    
    std::cout << "Generating " << VOCAB_SIZE << " N(0,10) logits with seed " << SEED << "..." << std::endl;
    
    std::vector<float> logits(VOCAB_SIZE);
    
    // LCG for random number generation
    unsigned int state = SEED;
    auto lcg = [&state]() -> float {
        state = (1664525u * state + 1013904223u);
        return (float)state / 4294967296.0f;
    };
    
    // Generate N(0, 10) distributed values
    for (int i = 0; i < VOCAB_SIZE; ++i) {
        float u1 = lcg();
        float u2 = lcg();
        u1 = std::max(u1, 1e-7f); // Avoid log(0)
        logits[i] = box_muller(u1, u2) * 10.0f; // N(0, 10)
    }
    
    // Save to binary file
    std::ofstream file("profile_logits_100k.bin", std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open file for writing!" << std::endl;
        return 1;
    }
    
    // Write metadata
    int32_t size = VOCAB_SIZE;
    file.write(reinterpret_cast<const char*>(&size), sizeof(int32_t));
    int32_t seed = SEED;
    file.write(reinterpret_cast<const char*>(&seed), sizeof(int32_t));
    
    // Write data
    file.write(reinterpret_cast<const char*>(logits.data()), VOCAB_SIZE * sizeof(float));
    file.close();
    
    std::cout << "Saved to profile_logits_100k.bin (" << (VOCAB_SIZE * sizeof(float) + 8) << " bytes)" << std::endl;
    
    // Print first 10 values for verification
    std::cout << "First 10 values: ";
    for (int i = 0; i < 10; ++i) {
        std::cout << logits[i] << " ";
    }
    std::cout << std::endl;
    
    return 0;
}
