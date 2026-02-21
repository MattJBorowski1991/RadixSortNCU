#include <cuda_runtime.h>
#include "../kernels/kernels.h"
#include "../utils/check_cuda.h"
#include <cstdio>
#include <string>
#include <cstring>
#include <map>
#include <cstdlib>
#include <ctime>
#include <vector>

int main(int argc, char** argv){


    int vocab_size = 32768;
    int num_batches = 256;
    int warmup_runs = 1;
    int runs = 2;
    std::string kernel = "radix_v1";


    for(int i = 1; i < argc; ++i){
        if (std::strncmp(argv[i], "--kernel=", 9) == 0) kernel = std::string(argv[i]+9);
        else if (std::strncmp(argv[i], "--vocab_size=", 13) == 0) vocab_size = std::atoi(argv[i]+13); 
        else if (std::strncmp(argv[i], "--num_batches=", 14) == 0) num_batches = std::atoi(argv[i]+14);
        else if (std::strncmp(argv[i], "--warmup_runs=", 14) == 0) warmup_runs = std::atoi(argv[i]+14);
        else if (std::strncmp(argv[i], "--runs=", 7) == 0) runs = std::atoi(argv[i]+7);
        else if (std::strcmp(argv[i], "--help") == 0){
            std::printf("Usage: %s --kernel=radix_v1|radix_v2|bitonic --vocab_size=131072 --num_batches=64 --warmup_runs=1 --runs=2", argv[0]);
            return 0;
        }
    }


    const int total_elements = num_batches * vocab_size;
    const int total_bytes = total_elements * sizeof(unsigned int);
    const size_t mem_required = 4 * total_bytes;
    std::printf("[main] Total elements: %lld, bytes per buffer: %lld, total required: %lld MB\n", 
               (long long)total_elements, (long long)total_bytes, (long long)(mem_required / 1024 / 1024));
    std::fflush(stdout);
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
    CHECK_CUDA(cudaMalloc(&d_input, total_bytes));
    CHECK_CUDA(cudaMalloc(&d_input_indices, total_bytes));
    CHECK_CUDA(cudaMalloc(&d_output, total_bytes));
    CHECK_CUDA(cudaMalloc(&d_output_indices, total_bytes));

    CHECK_CUDA(cudaMemcpy(d_input, h_input.data(), total_bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_input_indices, h_input_indices.data(), total_bytes, cudaMemcpyHostToDevice));

    using launcher_t = void(*)(unsigned int*, unsigned int*, unsigned int*, int, int);
    std::map<std::string, launcher_t> registry = {
        {"bitonic", solve_bitonic},
        {"radix_v1", solve_radix_v1},
        {"radix_v2", solve_radix_v2}
    };

    if(registry.find(kernel) == registry.end()){
        printf("Unknown kernel '%s' \n", kernel.c_str());
        return 1;
    }
    launcher_t launcher = registry[kernel];

    for(int w = 0; w < warmup_runs; ++w){
        std::printf("[warmup] Launching kernel '%s' (run %d)\n", kernel.c_str(), w);
        std::fflush(stdout);
        CHECK_CUDA(launcher(d_input, d_output, d_output_indices, vocab_size, num_batches));
        CHECK_CUDA(cudaDeviceSynchronize());
    }

    for(int r = 0; r < runs; ++r){
        std::printf("[run] Launching kernel '%s' (run %d)\n", kernel.c_str(), r);
        std::fflush(stdout);
        CHECK_CUDA(launcher(d_input, d_output, d_output_indices, vocab_size, num_batches));
        CHECK_CUDA(cudaDeviceSynchronize());
    }

    cudaFree(d_input);
    cudaFree(d_input_indices);
    cudaFree(d_output);
    cudaFree(d_output_indices);
    
    return 0;
}