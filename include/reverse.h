#pragma once

extern "C" void solve_reverse(
    unsigned int* asc_g_probs, 
    unsigned int* asc_indices, 
    float* desc_g_probs, 
    unsigned int* desc_indices,
    int vocab_size, int num_batches);
