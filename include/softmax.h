#pragma once

extern "C" void solve_softmax(const float* logits, float* g_probs, int vocab_size, int num_batches);

extern "C" void solve_normalize(const float* input, float* output, int N, const float* sums, int vocab_size, int num_batches);
