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
    bool verbose = false;

    // parse simple CLI flags: --vocab_size N (-v N), --num_batches B (-b B)
    for (int ai = 1; ai < argc; ++ai) {
        std::string s = argv[ai];
        if (s == "--vocab_size" || s == "-v") {
            if (ai + 1 < argc) vocab_size = std::atoi(argv[++ai]);
        } else if (s == "--num_batches" || s == "-b") {
            if (ai + 1 < argc) num_batches = std::atoi(argv[++ai]);
        } else if (s == "--verbose" || s == "-V" || s == "--print_all") {
            verbose = true;
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

    if (verbose) {
        std::cout << "Input data:" << std::endl;
        for (int b = 0; b < num_batches; ++b) {
            std::cout << "Batch " << b << ": ";
            for (int i = 0; i < vocab_size; ++i) {
                int idx = b * vocab_size + i;
                std::cout << h_input[idx] << " ";
            }
            std::cout << std::endl;
        }
    }

    cudaEvent_t start;
    cudaEvent_t stop;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    if (verbose) std::cout << "Testing Radix Sort..." << std::endl;
    cudaEventRecord(start);
    solve_radix(d_input, d_output, d_output_indices, vocab_size, num_batches);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float radix_ms = 0.0f;
    cudaEventElapsedTime(&radix_ms, start, stop);
    std::cout << "Radix duration: " << radix_ms << " ms" << std::endl;
    // Copy back
    cudaMemcpy(h_output.data(), d_output, total_elements * sizeof(unsigned int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_output_indices.data(), d_output_indices, total_elements * sizeof(unsigned int), cudaMemcpyDeviceToHost);

    // Check if sorted ascending
    bool radix_passed = true;
    for (int b = 0; b < num_batches; ++b) {
        if (verbose) std::cout << "Batch " << b << " (Radix): ";
        for (int i = 0; i < vocab_size; ++i) {
            int idx = b * vocab_size + i;
            if (verbose) std::cout << h_output[idx] << " ";
            if (i > 0 && h_output[idx] < h_output[idx - 1]) {
                radix_passed = false;
            }
        }
        if (verbose) std::cout << std::endl;
        if (verbose) {
            std::cout << "Indices: ";
            for (int i = 0; i < vocab_size; ++i) {
                int idx = b * vocab_size + i;
                std::cout << h_output_indices[idx] << " ";
            }
            std::cout << std::endl;
        }
    }
    if (!radix_passed) {
        std::cerr << "Radix sort failed: Not sorted!" << std::endl;
    }

    if (verbose) std::cout << "Testing Bitonic Sort..." << std::endl;
    // Reset input to original unsorted data
    cudaMemcpy(d_input, h_input.data(), total_elements * sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_input_indices, h_input_indices.data(), total_elements * sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaEventRecord(start);
    solve_bitonic(d_input, d_output, d_output_indices, vocab_size, num_batches);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float bitonic_ms = 0.0f;
    cudaEventElapsedTime(&bitonic_ms, start, stop);
    std::cout << "Bitonic duration: " << bitonic_ms << " ms" << std::endl;

    // Copy back
    cudaMemcpy(h_output.data(), d_output, total_elements * sizeof(unsigned int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_output_indices.data(), d_output_indices, total_elements * sizeof(unsigned int), cudaMemcpyDeviceToHost);

    // Check if sorted ascending
    bool bitonic_passed = true;
    for (int b = 0; b < num_batches; ++b) {
        if (verbose) std::cout << "Batch " << b << " (Bitonic): ";
        for (int i = 0; i < vocab_size; ++i) {
            int idx = b * vocab_size + i;
            if (verbose) std::cout << h_output[idx] << " ";
            if (i > 0 && h_output[idx] < h_output[idx - 1]) {
                bitonic_passed = false;
            }
        }
        if (verbose) std::cout << std::endl;
        if (verbose) {
            std::cout << "Indices: ";
            for (int i = 0; i < vocab_size; ++i) {
                int idx = b * vocab_size + i;
                std::cout << h_output_indices[idx] << " ";
            }
            std::cout << std::endl;
        }
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