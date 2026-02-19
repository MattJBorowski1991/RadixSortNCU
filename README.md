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


| Step | vocab_size=32,768 | vocab_size=131,072 | vocab_size=524,288 | vocab_size=1,048,576 |
|---|---:|---:|---:|---:|
| DtoD for buffer | 0.295232 ms <br><sub>0.30%</sub> | 1.16253 ms <br><sub>0.63%</sub> | 4.63226 ms <br><sub>0.89%</sub> | 9.26387 ms <br><sub>0.96%</sub> |
| init_indices | 0.452992 ms <br><sub>0.46%</sub> | 1.31686 ms <br><sub>0.71%</sub> | 4.81603 ms <br><sub>0.92%</sub> | 9.38662 ms <br><sub>0.97%</sub> |
| Total prefix_per_block | 7.55901 ms <br><sub>7.66%</sub> | 28.7225 ms <br><sub>15.56%</sub> | 113.474 ms <br><sub>21.76%</sub> | 228.477 ms <br><sub>23.59%</sub> |
| Total Loop over batches | 68.0684 ms <br><sub>68.94%</sub> | 70.526 ms <br><sub>38.21%</sub> | 79.3233 ms <br><sub>15.21%</sub> | 90.0407 ms <br><sub>9.29%</sub> |
| Total Memcpy HtoD for total_ones | 0.177504 ms <br><sub>0.18%</sub> | 0.181024 ms <br><sub>0.10%</sub> | 0.183264 ms <br><sub>0.04%</sub> | 0.19904 ms <br><sub>0.02%</sub> |
| Total radix_sort_asc_kernel | 18.9854 ms <br><sub>19.23%</sub> | 75.8713 ms <br><sub>41.10%</sub> | 304.002 ms <br><sub>58.29%</sub> | 606.85 ms <br><sub>62.64%</sub> |
| DtoD for output and cudaFrees | 3.1951 ms <br><sub>3.24%</sub> | 6.84032 ms <br><sub>3.70%</sub> | 15.0377 ms <br><sub>2.88%</sub> | 24.6282 ms <br><sub>2.54%</sub> |
| **Total** | **98.734238 ms** | **184.620244 ms** | **521.468554 ms** | **968.84596 ms** |


