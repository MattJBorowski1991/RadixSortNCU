#include <cuda_runtime.h>
#include "../kernels/kernels.h"
#include "../utils/cuda_utils.h"
#include <cstdio>
#include <string>
#include <cstring>
#include <map>
#include <time>
#include <vector>

int main(int argc, char** argv){


    int vocab_size = 131072;
    int num_batches = 64;
    int warmup_runs = 1;
    int runs = 2;
    std::string kernel = "radix_v1";


    for(int i = 1; i < argc; ++i){
        if (std::strncmp(argv[i], "--kernel=", 9) == 0) kernel = std::string(argv[i]+9);
        else if (std::strncmp(argv[i], "--vocab_size=", 11) == 0) vocab_size = std::atoi(argv[i]+12); 
        else if (std::strncmp(argv[i], "--num_batches=", 12) == 0) num_batches = std::atoi(argv[i]+12);
        else if (std::strncmp(argv[i], "--warmup_runs=", 12) == 0) warmup_runs = std::atoi(argv[i]+12);
        else if (std::strncmp(argv[i], "--runs=", 5) == 0) runs = std::atoi(argv[i]+5);
        else if (std::strncmp(argv[i], "--help") == 0){
            std::printf("Usage: %s --kernel=radix_v1 --vocab_size=131072 --num_batches=64 --warmup_runs=1 --runs=2", argv[0]);
            return 0;
        }
    }


    const int total_elements = num_batches * vocab_size;
    const int total_bytes = total_elements * sizeof(unsigned int);
    std::vector<unsigned int> h_input(total_elements);
    std::vector<unsigned int> h_input_indices(total_elements);
    srand(time(NULL));

    for(int b = 0; b < num_batches; ++b){
        for(int i = 0; i < vocab_size; ++i){
            int idx = b * vocab_size + i;
            h_input[idx] = rand() % 1000 + 1;
            h_input_indices[idx] = idx;
        }
    }

    unsigned int *d_input, *d_input_indices, *d_output, *d_output_indices;
    cudaMalloc(&d_input, total_bytes);
    cudaMalloc(&d_input_indices, total_bytes);
    cudaMalloc(&d_output, total_bytes);
    cudaMalloc(&d_output_indices, total_bytes);

    cudaMemcpy(d_input, h_input.data(), total_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_input_indices, h_input_indices.data(), total_bytes, cudaMemcpyHostToDevice);

    using launcher_t = void(*)(unsigned int*, unsigned int*, unsigned int*, int, int);
    std::map<std::string, launcher_t> registry = {
        {"bitonic", solve_bitonic},
        {"radix_v1", solve_radix_v1}
    };

    if(registry.find(kernel) == registry.end()){
        printf("Unknown kernel '%s' \n", kernel.c_str());
        return 1;
    }
    launcher_t launcher = registry[kernel];

    for(int w = 0; w < warmup_runs; ++w){
        CHECK_CUDA(launcher(d_input, d_output, d_output_indices, vocab_size, num_batches));
    }

    for(int r = 0; r < runs; ++r){
        CHECK_CUDA(launcher(d_input, d_output, d_output_indices, vocab_size, num_batches));
    }


    cudaFree(d_input);
    cudaFree(d_input_indices);
    cudaFree(d_output);
    cudaFree(d_output_indices);
    
    return 0;
}