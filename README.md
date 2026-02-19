# RadixSortNCU

## Set-up

**Compile**

```bash
make clean && make NVCC_ARCH=XX  
```

**Smoke-run**

```bash
./bin/runner --kernel=radix_v1 --vocab_size=32768 --num_batches=1 --warmup_runs=1 --runs=1
```

**Quick Profile Run**

```bash
ncu ./bin/profile_harness --num_batches=256 --vocab_size=32768 --kernel=radix_v1 --warmup_runs=0 --runs=1 > prof/txt/radix_v1_256_32768.txt
```

**Nvprof Profile**

```bash
nvprof ./bin/profile_harness --num_batches=256 --vocab_size=32768 --kernel=radix_v1 --warmup_runs=0 --runs=1
```

or for per-call detail:

```bash
nvprof --print-gpu-trace ./bin/profile_harness --num_batches=256 --vocab_size=32768 --kernel=radix_v1 --warmup_runs=0 --runs=1 2>&1 | tee prof/txt/nvprof_gputrace_radix_v1_256_32768.txt
```

**Nsight Compute Profile**

```bash
ncu --import-source yes --set full --export profiles/ncu/radix_v1.ncu-rep ./bin/profile_harness --kernel=radix_v1 --warmup_runs=1 --runs=2
```


## Runs

**Run 1**

We set number of batches to 256 and conduct basing timings via Nvprof and cudaEvents.

The GPU trace outputs in `prof/txt` reveal that small transfers within the per-bit loop in the radix kernel are approximately 3-3.5x faster for HtoD compared to DtoH.

### Nvprof Summary

Table below shows `Time(%) / Time` for the main GPU activities extracted from the nvprof outputs in `prof/txt/` for `--num_batches=256` and four `--vocab_size` values.

| Op \ vocab | 32,768 | 131,072 | 524,288 | 1,048,576 |
|---|---:|---:|---:|---:|
| `radix_sort_asc_kernel` | 35.0% <br><sub>10 ms</sub> | 42.1% <br><sub>38 ms</sub> | 44.3% <br><sub>154 ms</sub> | 44.2% <br><sub>308 ms</sub> |
| `[cudaMemcpy HtoD]` | 32.1% <br><sub>9 ms</sub> | 32.8% <br><sub>30 ms</sub> | 33.2% <br><sub>115 ms</sub> | 33.8% <br><sub>235 ms</sub> |
| `[cudaMemcpy DtoH]` | 15.8% <br><sub>4 ms</sub> | 4.9% <br><sub>4 ms</sub> | 1.3% <br><sub>5 ms</sub> | 0.7% <br><sub>5 ms</sub> |
| `prefix_per_block` | 15.2% <br><sub>4 ms</sub> | 17.8% <br><sub>16 ms</sub> | 18.5% <br><sub>64 ms</sub> | 18.8% <br><sub>131 ms</sub> |
| `[cudaMemcpy DtoD]` | 1.5% <br><sub>0 ms</sub> | 1.9% <br><sub>2 ms</sub> | 1.9% <br><sub>7 ms</sub> | 1.9% <br><sub>14 ms</sub> |
| `init_indices` | 0.5% <br><sub>0 ms</sub> | 0.6% <br><sub>1 ms</sub> | 0.7% <br><sub>2 ms</sub> | 0.7% <br><sub>5 ms</sub> |
| **Latency (ms)** | **27 ms** | **91 ms** | **347 ms** | **697 ms** |
| **ns / token** | **1673 ns** | **1391 ns** | **1325 ns** | **1328 ns** |

### cudaEvent Timings Summary

Given the surprisingly high HtoD overhead in Nvprof summary, I investigated further with isolated cudaEvent timings for all the steps in the code:

| Op (%) \ vocab_size | 32,768 | 131,072 | 524,288 | 1,048,576 |
|---|---:|---:|---:|---:|
| DtoD for buffer | 0.3% <br><sub>0 ms</sub> | 0.6% <br><sub>1 ms</sub> | 0.9% <br><sub>2 ms</sub> | 1.0% <br><sub>5 ms</sub> |
| init_indices | 0.5% <br><sub>0 ms</sub> | 0.7% <br><sub>1 ms</sub> | 0.9% <br><sub>2 ms</sub> | 1.0% <br><sub>5 ms</sub> |
| In per-bit loop: prefix_per_block | 7.7% <br><sub>4 ms</sub> | 15.6% <br><sub>14 ms</sub> | 21.8% <br><sub>57 ms</sub> | 23.6% <br><sub>114 ms</sub> |
| In per-bit loop: loop over batches* | 68.9% <br><sub>34 ms</sub> | 38.2% <br><sub>35 ms</sub> | 15.2% <br><sub>40 ms</sub> | 9.3% <br><sub>45 ms</sub> |
| In per-bit loop: HtoD for total_ones | 0.2% <br><sub>0 ms</sub> | 0.1% <br><sub>0 ms</sub> | 0.0% <br><sub>0 ms</sub> | 0.0% <br><sub>0 ms</sub> |
| In per-bit loop: radix | 19.2% <br><sub>9 ms</sub> | 41.1% <br><sub>38 ms</sub> | 58.3% <br><sub>152 ms</sub> | 62.6% <br><sub>303 ms</sub> |
| DtoD for output and cudaFrees | 3.2% <br><sub>2 ms</sub> | 3.7% <br><sub>3 ms</sub> | 2.9% <br><sub>8 ms</sub> | 2.5% <br><sub>12 ms</sub> |
| **Latency (ms)** | **49 ms** | **92 ms** | **261 ms** | **484 ms** |
| **ns / token** | **1495 ns** | **701 ns** | **498 ns** | **462 ns** |

*includes: DtoH + loop over batches + HtoD

What we can see is the overhead from Nvprof (vs cudaEventRecord) is substantial and that it only profiles GPU activities - on-host computing is not included. 

The timings also confirms that:
1. The on-host nested loop (bits->batches->blocks) is a major bottleneck for small to medium vocabs and goes down to 9% for the largest vocab.
2. The two kernels used - radix and prefix_per_block become similarly larger bottlenecks (from 27% to 86%) for the largest vocab size.
3. The loop from point 1 and the two kernels from points two account for almost all the latency. The remaining part of the code (memory transfers, init_indices) constitutes c.a. 4.0-4.5% depending on vocab.

### Why we focus on Radix and not Bitonic

Below are the equivalent latencies for Bitonic - a reason why we focus on profiling and optimizing Radix.

| Vocab Size | Radix | Bitonic | x |
|---|---:|---:|---:|
| 32,768 | **49 ms** | **77.5 ms** | **1.6x** |
| 131,072 | **92 ms** | **356.3 ms** | **3.9x** |
| 524,288 | **261 ms** | **1642 ms** | **6.3x** |
| 1,048,576 | **484 ms** | **3500 ms** | **7.25x** |



**Run 2**

We profile the two kernels (in isolation): prefix_per_block and radix.
