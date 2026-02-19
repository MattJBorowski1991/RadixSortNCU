# RadixSortNCU

## Compile

```bash
make clean && make NVCC_ARCH=XX  
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


**Profiling Summary**

Table below shows `Time(%) / Time` for the main GPU activities extracted from the nvprof outputs in `prof/txt/` for `--num_batches=256` and four `--vocab_size` values.

| Activity | vocab=32,768 | vocab=131,072 | vocab=524,288 | vocab=1,048,576 |
|---|---:|---:|---:|---:|
| `radix_sort_asc_kernel` | 34.89% <br><sub>19.131 ms</sub> | 42.07% <br><sub>76.715 ms</sub> | 44.33% <br><sub>307.95 ms</sub> | 44.17% <br><sub>615.30 ms</sub> |
| `[cudaMemcpy HtoD]` | 32.08% <br><sub>17.586 ms</sub> | 32.75% <br><sub>59.726 ms</sub> | 33.20% <br><sub>230.59 ms</sub> | 33.75% <br><sub>470.20 ms</sub> |
| `[cudaMemcpy DtoH]` | 15.83% <br><sub>8.6769 ms</sub> | 4.91% <br><sub>8.9485 ms</sub> | 1.34% <br><sub>9.3332 ms</sub> | 0.72% <br><sub>10.085 ms</sub> |
| `prefix_per_block` | 15.17% <br><sub>8.3159 ms</sub> | 17.79% <br><sub>32.441 ms</sub> | 18.52% <br><sub>128.65 ms</sub> | 18.76% <br><sub>261.33 ms</sub> |
| `[cudaMemcpy DtoD]` | 1.51% <br><sub>0.82866 ms</sub> | 1.85% <br><sub>3.3683 ms</sub> | 1.94% <br><sub>13.498 ms</sub> | 1.94% <br><sub>27.053 ms</sub> |
| `init_indices` | 0.53% <br><sub>0.28854 ms</sub> | 0.63% <br><sub>1.1464 ms</sub> | 0.66% <br><sub>4.5893 ms</sub> | 0.66% <br><sub>9.1829 ms</sub> |
| **Latency (ms)** | **54.827 ms** | **182.345 ms** | **694.611 ms** | **1393.151 ms** |
| **ns / token** | **1673 ns** | **1391 ns** | **1325 ns** | **1328 ns** |

Notes:
- Values taken from `prof/txt/run1/nvprof_radix_v1_256_*.txt` (nvprof "GPU activities" section).
- Percent/time comparisons show that as `vocab_size` increases the sort kernel and HtoD transfers dominate total GPU activity.

### Sequence of Execution in the Kernel

Given the surprisingly high HtoD overhead I investigate further with isolated cudaEvent timings for all the steps in the code:


| Op (ms) \ vocab_size | 32,768 | 131,072 | 524,288 | 1,048,576 |
|---|---:|---:|---:|---:|
| DtoD for buffer | 0.30 <br><sub>0.30%</sub> | 1.16 <br><sub>0.63%</sub> | 4.63 <br><sub>0.89%</sub> | 9.26 <br><sub>0.96%</sub> |
| init_indices | 0.45 <br><sub>0.46%</sub> | 1.32 <br><sub>0.71%</sub> | 4.82 <br><sub>0.92%</sub> | 9.39 <br><sub>0.97%</sub> |
| Total prefix_per_block | 7.56 <br><sub>7.66%</sub> | 28.72 <br><sub>15.56%</sub> | 113.47 <br><sub>21.76%</sub> | 228.48 <br><sub>23.59%</sub> |
| Total Loop over batches | 68.07 <br><sub>68.94%</sub> | 70.53 <br><sub>38.21%</sub> | 79.32 <br><sub>15.21%</sub> | 90.04 <br><sub>9.29%</sub> |
| Total Memcpy HtoD for total_ones | 0.18 <br><sub>0.18%</sub> | 0.18 <br><sub>0.10%</sub> | 0.18 <br><sub>0.04%</sub> | 0.20 <br><sub>0.02%</sub> |
| Total radix_sort_asc_kernel | 18.99 <br><sub>19.23%</sub> | 75.87 <br><sub>41.10%</sub> | 304.00 <br><sub>58.29%</sub> | 606.85 <br><sub>62.64%</sub> |
| DtoD for output and cudaFrees | 3.20 <br><sub>3.24%</sub> | 6.84 <br><sub>3.70%</sub> | 15.04 <br><sub>2.88%</sub> | 24.63 <br><sub>2.54%</sub> |
| **Total** | **98.73** | **184.62** | **521.47** | **968.85** |


