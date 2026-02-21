
## Run 1: Radix Sort Launcher Analysis

We profile the complete radix sort launcher with 256 batches using both Nvprof and cudaEvent instrumentation to understand the contribution of each component.

### Context

The GPU trace outputs in `prof/txt` reveal that small transfers within the per-bit loop in the radix kernel are approximately 3-3.5x faster for HtoD compared to DtoH.

### Nvprof Summary

Table below shows `Time(%) / Time` for the main GPU activities extracted from the nvprof outputs in `prof/txt/` for `--num_batches=256` and four `--vocab_size` values.

| **Activity** | **vocab=32,768** | **vocab=131,072** | **vocab=524,288** | **vocab=1,048,576** |
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

Given the surprisingly high HtoD overhead in Nvprof, we investigated further with isolated cudaEvent timings for all execution stages:

| **Op (%) \ vocab_size** | **32,768** | **131,072** | **524,288** | **1,048,576** |
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

*includes: DtoH + loop over batches + HtoD*

#### Analysis & Key Insights

**Nvprof vs. cudaEventRecord Overhead**

The overhead from Nvprof (vs cudaEventRecord) is substantial, accounting for a ~44% increase in total latency. This discrepancy occurs because Nvprof profiles GPU activities across the entire driver (in `drivers/main.cu`) and includes implicit memory operations, whereas cudaEventRecord measures only the launcher itself. Detailed GPU traces in `/prof/txt/run1/nvprof_gputrace_radix_v1_256_1048576.txt` reveal two large HtoD transfers of ~230 ms each (~460 ms total) occurring before the DtoD buffer copy—these driver-level operations are captured by Nvprof but not by our cudaEvent instrumentation.

**Bottleneck Components**

The cudaEventRecord timings confirm three critical findings:

1. **CPU-side loop overhead**: The on-host nested loop (bits→batches→blocks) dominates small-to-medium vocabularies (69% at 32K) but diminishes to 9% at 1M tokens as GPU kernel costs rise.

2. **GPU kernel scaling**: The radix and prefix_per_block kernels together grow from 27% at 32K tokens to **86% at 1M tokens**, becoming the dominant bottleneck at larger scales.

3. **Remaining overhead**: Memory transfers and index initialization account for 4.0–4.5% across all vocabulary sizes and are not optimization targets.

**Code-Level Attribution**

The distribution of latency across code regions for the largest vocabulary (1M tokens):

![Radix_v1 code with latency %](../images/run1/radix_v1_code_prof_1048576.png)

#### Timing breakdown (cudaEventRecord)

The chart below shows the percent contribution of each pipeline component (mem/init, prefix kernel, on-host loop, radix kernel) for four vocabulary sizes. Group annotations show the measured total latency per configuration.

![Radix sort - cudaEventRecord timings](../images/run1/radix_event_timing_chart.png)

#### Why Radix Sort?

Radix sort significantly outperforms Bitonic sort across all vocabulary sizes, justifying our focus on radix optimization.

| **Vocab Size** | **Radix** | **Bitonic** | **Advantage (Bitonic / Radix)** |
|---|---:|---:|---:|
| 32,768 | **49 ms** | **77.5 ms** | **1.6x** |
| 131,072 | **92 ms** | **356.3 ms** | **3.9x** |
| 524,288 | **261 ms** | **1642 ms** | **6.3x** |
| 1,048,576 | **484 ms** | **3500 ms** | **7.25x** |