#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cassert>
#include <cstdlib>
#include <ctime>
#include <string>

// Declare the functions (assuming they are compiled separately)
extern "C" void solve_radix(
    unsigned int *g_probs,
    unsigned int *asc_g_probs, unsigned int* asc_indices,
    int vocab_size, int num_batches);

extern "C" void solve_bitonic(
    unsigned int *input,
    unsigned int *output, unsigned int *output_indices,
    int vocab_size, int num_batches);

int main(int argc, char** argv) {
    int vocab_size = 4;
    int num_batches = 1;

    // parse simple CLI flags: --vocab_size N (-v N), --num_batches B (-b B)
    for (int ai = 1; ai < argc; ++ai) {
        std::string s = argv[ai];
        if (s == "--vocab_size" || s == "-v") {
            if (ai + 1 < argc) vocab_size = std::atoi(argv[++ai]);
        } else if (s == "--num_batches" || s == "-b") {
            if (ai + 1 < argc) num_batches = std::atoi(argv[++ai]);
        } else if (s == "--help" || s == "-h") {
            std::cout << "Usage: " << argv[0] << " [--vocab_size N] [--num_batches B]\n";
            return 0;
        }
    }

    // if (vocab_size % 32 != 0) {
    //     std::cerr << "vocab_size must be a multiple of 32\n";
    //     return 1;
    // }

    const int total_elements = vocab_size * num_batches;

    srand(time(NULL));
    cudaDeviceReset();  // Reset GPU to free any previous allocations
    // Host data
    std::vector<unsigned int> h_input(total_elements);
    std::vector<unsigned int> h_input_indices(total_elements);
    std::vector<unsigned int> h_output(total_elements);
    std::vector<unsigned int> h_output_indices(total_elements);

    // Fill with random test data: values from 1 to 1000
    for (int b = 0; b < num_batches; ++b) {
        for (int i = 0; i < vocab_size; ++i) {
            int idx = b * vocab_size + i;
            h_input[idx] = rand() % 1000 + 1;
            h_input_indices[idx] = idx;
        }
    }

    // Device pointers
    unsigned int *d_input, *d_input_indices, *d_output, *d_output_indices;
    cudaMalloc(&d_input, total_elements * sizeof(unsigned int));
    cudaMalloc(&d_input_indices, total_elements * sizeof(unsigned int));
    cudaMalloc(&d_output, total_elements * sizeof(unsigned int));
    cudaMalloc(&d_output_indices, total_elements * sizeof(unsigned int));

    // Copy to device
    cudaMemcpy(d_input, h_input.data(), total_elements * sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_input_indices, h_input_indices.data(), total_elements * sizeof(unsigned int), cudaMemcpyHostToDevice);

    std::cout << "Input data:" << std::endl;
    for (int b = 0; b < num_batches; ++b) {
        std::cout << "Batch " << b << ": ";
        for (int i = 0; i < vocab_size; ++i) {
            int idx = b * vocab_size + i;
            std::cout << h_input[idx] << " ";
        }
        std::cout << std::endl;
    }

    std::cout << "Testing Radix Sort..." << std::endl;
    solve_radix(d_input, d_output, d_output_indices, vocab_size, num_batches);

    // Copy back
    cudaMemcpy(h_output.data(), d_output, total_elements * sizeof(unsigned int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_output_indices.data(), d_output_indices, total_elements * sizeof(unsigned int), cudaMemcpyDeviceToHost);

    // Check if sorted ascending
    bool radix_passed = true;
    for (int b = 0; b < num_batches; ++b) {
        std::cout << "Batch " << b << " (Radix): ";
        for (int i = 0; i < vocab_size; ++i) {
            int idx = b * vocab_size + i;
            std::cout << h_output[idx] << " ";
            if (i > 0 && h_output[idx] < h_output[idx - 1]) {
                radix_passed = false;
            }
        }
        std::cout << std::endl;
        std::cout << "Indices: ";
        for (int i = 0; i < vocab_size; ++i) {
            int idx = b * vocab_size + i;
            std::cout << h_output_indices[idx] << " ";
        }
        std::cout << std::endl;
    }
    if (!radix_passed) {
        std::cerr << "Radix sort failed: Not sorted!" << std::endl;
    }

    std::cout << "Testing Bitonic Sort..." << std::endl;
    // Reset input to original unsorted data
    cudaMemcpy(d_input, h_input.data(), total_elements * sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_input_indices, h_input_indices.data(), total_elements * sizeof(unsigned int), cudaMemcpyHostToDevice);
    solve_bitonic(d_input, d_output, d_output_indices, vocab_size, num_batches);

    // Copy back
    cudaMemcpy(h_output.data(), d_output, total_elements * sizeof(unsigned int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_output_indices.data(), d_output_indices, total_elements * sizeof(unsigned int), cudaMemcpyDeviceToHost);

    // Check if sorted ascending
    bool bitonic_passed = true;
    for (int b = 0; b < num_batches; ++b) {
        std::cout << "Batch " << b << " (Bitonic): ";
        for (int i = 0; i < vocab_size; ++i) {
            int idx = b * vocab_size + i;
            std::cout << h_output[idx] << " ";
            if (i > 0 && h_output[idx] < h_output[idx - 1]) {
                bitonic_passed = false;
            }
        }
        std::cout << std::endl;
        std::cout << "Indices: ";
        for (int i = 0; i < vocab_size; ++i) {
            int idx = b * vocab_size + i;
            std::cout << h_output_indices[idx] << " ";
        }
        std::cout << std::endl;
    }
    if (!bitonic_passed) {
        std::cerr << "Bitonic sort failed: Not sorted!" << std::endl;
    }

    if (radix_passed && bitonic_passed) {
        std::cout << "All tests passed!" << std::endl;
    } else {
        std::cout << "Some tests failed!" << std::endl;
    }

    // Cleanup
    cudaFree(d_input);
    cudaFree(d_input_indices);
    cudaFree(d_output);
    cudaFree(d_output_indices);

    return 0;
}