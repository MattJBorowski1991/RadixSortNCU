# RadixSortNCU

## Compile

```bash
nvcc -std=c++17 -O3 -Iinclude -Ikernels -Iutils drivers/main.cu kernels/radix_v1.cu kernels/bitonic.cu -o bin/profile_harness
```

## Smoke-run

```bash
./bin/runner --kernel=radix_v1 --vocab_size=32768 --num_batches=1 --warmup_runs=1 --runs=1
```

## Quick Profile Run

```bash
ncu ./bin/profile_harness --num_batches=256 --vocab_size=32768 --kernel=radix_v1 --warmup_runs=0 --runs=1 > prof/txt/radix_v1_256_32768.txt
```

## Profile with nvprof

```bash
nvprof ./bin/profile_harness --num_batches=256 --vocab_size=32768 --kernel=radix_v1 --warmup_runs=0 --runs=1
```

or for per-call detail:

```bash
nvprof --print-gpu-trace ./bin/profile_harness --num_batches=256 --vocab_size=32768 --kernel=radix_v1 --warmup_runs=0 --runs=1 2>&1 | tee prof/txt/nvprof_gputrace_radix_v1_256_32768.txt
```

## Profile with Nsight Compute

```bash
ncu --import-source yes --set full --export profiles/ncu/radix_v1.ncu-rep ./bin/profile_harness --kernel=radix_v1 --warmup_runs=1 --runs=2
```


## Run 1

The GPU trace outputs in `prof/txt` reveal the following insights:

- Small transfers within the per-bit loop in the radix kernel are approximately 3-3.5x faster for HtoD compared to DtoH.

### Sequence of Execution in the Kernel

1. DtoD of `total_elements = num_batches * vocab_size`
2. 2 x `init_indices` kernel
3. For loop over 32 bits:
   1. `prefix_per_block` kernel
   2. For loop over batches:
      1. DtoH of block sums (of 1s)
      2. For loop to calculate exclusive prefix sum (of 1s)
      3. HtoD of exclusive prefix sum per block
   3. HtoD of total sum (of 1s) per batch
   4. `radix_sort_asc`

**Profiling Summary**

Table below shows `Time(%) / Time` for the main GPU activities extracted from the nvprof outputs in `prof/txt/` for `--num_batches=256` and four `--vocab_size` values.

| Activity | vocab=32,768 | vocab=131,072 | vocab=524,288 | vocab=1,048,576 |
|---|---:|---:|---:|---:|
| `radix_sort_asc_kernel` | 34.89% / 19.131 ms | 42.07% / 76.715 ms | 44.33% / 307.95 ms | 44.17% / 615.30 ms |
| `[cudaMemcpy HtoD]` | 32.08% / 17.586 ms | 32.75% / 59.726 ms | 33.20% / 230.59 ms | 33.75% / 470.20 ms |
| `[cudaMemcpy DtoH]` | 15.83% / 8.6769 ms | 4.91% / 8.9485 ms | 1.34% / 9.3332 ms | 0.72% / 10.085 ms |
| `prefix_per_block` | 15.17% / 8.3159 ms | 17.79% / 32.441 ms | 18.52% / 128.65 ms | 18.76% / 261.33 ms |
| `[cudaMemcpy DtoD]` | 1.51% / 0.82866 ms | 1.85% / 3.3683 ms | 1.94% / 13.498 ms | 1.94% / 27.053 ms |
| `init_indices` | 0.53% / 0.28854 ms | 0.63% / 1.1464 ms | 0.66% / 4.5893 ms | 0.66% / 9.1829 ms |
| **Latency (ms)** | **54.827 ms** | **182.345 ms** | **694.611 ms** | **1393.151 ms** |
| **ns / token** | **1673 ns** | **1391 ns** | **1325 ns** | **1328 ns** |

Notes:
- Values taken from `prof/txt/nvprof_radix_v1_256_*.txt` (nvprof "GPU activities" section).
- Percent/time comparisons show that as `vocab_size` increases the sort kernel and HtoD transfers dominate total GPU activity.

