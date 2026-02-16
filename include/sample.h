#pragma once

extern "C" void solve_sample(const float* d_p_n_nucleus, const int* seed, int* sampled_token, int max_nucleus_size, const unsigned int* desc_indices, int vocab_size, int num_batches);