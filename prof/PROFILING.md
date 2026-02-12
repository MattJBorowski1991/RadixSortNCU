# Top-P Sampling Profiling

This directory contains performance profiling tools for the CUDA top-p sampling implementation.

## Setup & Run Profiling

### Quick Start
```bash
chmod +x generate_all_profile_data.sh setup_profiling.sh
./generate_all_profile_data.sh
./setup_profiling.sh

# Run profiling
./profile                           # 100k vocab (default)
./profile profile_logits_1m.bin     # 1M vocab (large scale)
```

This will:
1. Generate 100k and 1M N(0,10) distributed logits (realistic LLM vocab sizes)
2. Compile the CUDA code (test and profile executables separately)
3. Run profiling and print timing results for each kernel

## Profile Datasets

Two pre-generated datasets available:

| Dataset | File | Vocab Size | File Size | Use Case |
|---------|------|-----------|-----------|----------|
| 100k | `profile_logits_100k.bin` | 100,000 | 400 KB | Standard LLM vocab (GPT-3 range) |
| 1M | `profile_logits_1m.bin` | 1,000,000 | 4 MB | Large vocab models, scaling analysis |

Both contain N(0, 10) distributed values (mean=0, std=10) generated with seed=42 for reproducibility.

## Output Format

The profiling output shows for each step:
- **Time (ms)**: Total execution time
- **Read (GB)**: Data read from global memory
- **Read (GB/s)**: Memory bandwidth for reads
- **Write (GB)**: Data written to global memory
- **Write (GB/s)**: Memory bandwidth for writes
- **GFLOPS**: Floating-point operations per second

## Steps Profiled

1. **STEP 1: Softmax** - Computing probabilities from logits
2. **STEP 2: Sort** - Radix sorting probabilities in descending order
3. **STEP 3.1: Nucleus Select** - Finding nucleus cutoff via prefix sum
4. **STEP 3.2: Normalize** - Normalizing nucleus probabilities
5. **STEP 3.3: Prefix Sum** - Computing prefix sum for sampling
6. **STEP 4: Sample** - Selecting token via LCG sampling

## Files

- `generate_profile_data.cpp` - Generates 100k N(0,10) logits
- `generate_profile_data_1m.cpp` - Generates 1M N(0,10) logits
- `generate_all_profile_data.sh` - Generates both datasets
- `profile_logits_100k.bin` - Pre-generated 100k test data
- `profile_logits_1m.bin` - Pre-generated 1M test data
- `profile.cu` - Standalone profiling executable
- `profile` - Compiled profiling executable

## Usage

### Generate Data
```bash
# Generate both 100k and 1M datasets
chmod +x generate_all_profile_data.sh
./generate_all_profile_data.sh

# Or generate individually
g++ -O2 generate_profile_data.cpp -o gen_data_100k -lm && ./gen_data_100k
g++ -O2 generate_profile_data_1m.cpp -o gen_data_1m -lm && ./gen_data_1m
```

### Compile
```bash
# Compile profiling executable only
nvcc profile.cu top_p_sampling.cu -o profile -std=c++14 -O3
```

### Profile
```bash
# Profile with 100k vocab
./profile

# Profile with 1M vocab
./profile profile_logits_1m.bin
```

## Scaling Analysis

Run both datasets to analyze scaling:
```bash
echo "=== 100k Vocab ===" && ./profile
echo ""
echo "=== 1M Vocab ===" && ./profile profile_logits_1m.bin
```

Compare the timing/throughput to understand:
- How memory bandwidth utilization changes with vocab size
- Scalability of sort kernel (typically scales sub-linearly)
- Impact on overall latency

## Notes

- Fixed data ensures reproducible profiling runs
- 100k vocab is realistic for standard LLMs
- 1M vocab for large-scale models and bottleneck analysis
- Separate `profile` executable keeps profiling isolated from tests
- Profile data is deterministic (seed=42) for consistent results
