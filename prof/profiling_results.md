++# Profiling Results

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

### Analysis & Key Insights

**Nvprof vs. cudaEventRecord Overhead**

## Run 2: Individual Kernel Profiling

We profile the two primary kernels in isolation: `prefix_per_block` and `radix_sort_asc_kernel`.

### I.1. `radix_v1`

**The major bottlenecks for this kernel were:**

- **Long Scoreboard Stalls** — *Est. Speedup: 44.13%*
	- On average, each warp spends **11.1 cycles** stalled waiting for a scoreboard dependency on an L1TEX operation. This stall type represents about **44.1%** of the average 25.1 cycles between issuing two instructions.
	- Guidance: identify the producing instruction, improve memory access patterns (coalescing), increase data locality or change cache config, and consider moving frequently used data to shared memory.
	- Metrics:
		- `smsp__issue_active.avg.per_cycle_active = 0.301733` — increase issued instructions per cycle
		- `smsp_average_long_scoreboard = 11.0839` — reduce long scoreboard cycles

- **Uncoalesced Global Accesses** — *Est. Speedup: 13.48%*
	- Kernel shows uncoalesced global reads/writes resulting in **917,186 excessive sectors** (≈14% of 6,684,354 sectors).
	- Guidance: check the L2 theoretical sectors table and refactor memory layout / access stride.

- **L1TEX Global Store Access Pattern** — *Est. Speedup: 13.41%*
	- Global store pattern uses **≈22.3 of 32 bytes per sector** on average; suggests a stride/aliasing issue.

**Visual evidence**

![Radix v1 - Stall Long Scoreboard](prof/images/run2/run2_radix_v1_long_scoreboard.png)

Long Scoreboard & Barrier stalls account for the majority of stalls:

![Radix v1 - Source Counters](prof/images/run2/run2_radix_v1_source_counters.png)

Memory workload (SRAM usage vs L1/L2 hit rates):

![Radix v1 - Memory Workload](prof/images/run2/run2_radix_v1_mem_workload.png)

The single source line producing the largest stalls is shown below (global-to-global load/store):

![Radix v1 - Source Code](prof/images/run2/run2_radix_v1_source_1.png)

### I.2. `prefix_per_block`

**Primary bottleneck reported by NCU:**

- **L1TEX Global Store Access Pattern** — *Est. Speedup: 32.89%*
	- On average only **4.0 of 32 bytes per sector** are utilized by each thread (very sparse writes).
	- Metric: `smsp__sass_average_data_bytes_per_sector_mem_global_op_st.ratio = 4`
	- Guidance: increase bytes utilized per sector (coalesce, restructure writes).

Memory workload indicates near-zero L1/L2 hit rates for this kernel, so L1 throughput is not meaningful here:

![Radix v1 - prefix_per_block - L1](prof/images/run2/run2_prefix_l1tex_store.png)

Source counters flagged the following source locations as uncoalesced contributors:

![Radix v1 - prefix_per_block - source 1](prof/images/run2/run2_prefix_source_1.png)

![Radix v1 - prefix_per_block - source 2](prof/images/run2/run2_prefix_source_2.png)

These are algorithmic: the tree reduction produces a single output per block (one thread writes `block_sums`), creating a sparse scatter pattern and the low 4/32 utilization.

The line responsible for major stalls in both `radix_v1` and this kernel is:

```c
val = 1 - ((input_batch[gid] >> bit) & 1);
```

**Corollary:** use of Shared Memory was proposed to reduce the observed stalls and improve locality.

---

## II. `radix_v2.cu`

**TL;DR:** duration of `radix` remained unchanged; duration of `prefix` increased by **~20%** (deterioration). Moving inputs to SRAM did not help overall.

### II.1. `radix_v2`

- Remaining significant bottleneck: **L1TEX Global Store Access Pattern** (est. speedup ≈ **+10%**, improved from +33%).

- The previous dominant `Stall Long Scoreboard` was reduced by more than half, however other stall categories increased (Stall Barrier, Stall Short Scoreboard, Stall MIO Throttle). Overall Warp Cycles per Issued Instruction decreased by ~10%.

![Radix v2 - Memory Workload](prof/images/run2/run2_radix_v2_long_scoreboard.png)

Memory workload shows heavier SRAM usage, at the expense of L1 (L1 hit rate down **27% → 11%**):

![Radix v2 - Memory Workload](prof/images/run2/run2_radix_v2_mem_workload.png)

### II.2. `prefix`

The kernel performance deteriorated; investigation notes:

- **L1TEX Global Store Access Pattern** remains the top issue (est. speedup now **+37%**, up from +32%).
- This bottleneck is algorithmic (sparse per-block writes) and requires redesign to eliminate.

Effects observed after moving `input_batch` to SRAM:

- Branch instructions increased **+65%** (to ~1.33M).
- All active cycle counts (Total Elapsed & Average Active) increased **~20%**, except **Average L2** and **Average DRAM** active cycles which remained unchanged.

The remaining hotspot now attributes most cost to the SRAM load line:

```c
s_input_batch[tid] = __ldg(input_batch + gid);
```

**Corollary:** try a different approach (redesign algorithm, reduce uncoalesced writes, or compute offsets entirely on GPU).
