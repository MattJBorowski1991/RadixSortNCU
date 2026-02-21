#ifndef KERNELS_H
#define KERNELS_H

extern "C" void solve_radix_v1(unsigned int *input, 
    unsigned int *output, unsigned int* output_indices, 
    int vocab_size, int num_batches);

extern "C" void solve_radix_v2(unsigned int *input, 
    unsigned int *output, unsigned int* output_indices, 
    int vocab_size, int num_batches);

extern "C" void solve_radix_v3(unsigned int *input, 
    unsigned int *output, unsigned int* output_indices, 
    int vocab_size, int num_batches);

extern "C" void solve_bitonic(unsigned int *input,
    unsigned int *output, unsigned int *output_indices, 
    int vocab_size, int num_batches);

#endif