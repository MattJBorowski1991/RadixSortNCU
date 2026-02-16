#pragma once

extern "C" void solve_nucleus_part1(const float* desc_g_probs, int* h_nucleus_sizes, float* h_global_probs_sums, int* max_nucleus_size, int vocab_size, int num_batches, const float* p);

extern "C" void solve_nucleus_part2(const float* desc_g_probs, int* h_nucleus_sizes, float* d_nucleus, int max_nucleus_size, int vocab_size, int num_batches);
