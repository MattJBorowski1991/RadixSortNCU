# RadixSortNCU

This project profiles and optimizes the **Radix Sort** kernel pipeline for GPU-accelerated top-p sampling.

## Overview

The analysis reveals that sorting becomes the primary bottleneck in large-vocabulary pipelines, accounting for up to **86%** of latency at vocabulary sizes of 1M tokens. Through systematic profiling of the complete radix sort launcher—including both GPU kernels and CPU-side orchestration—we identify and measure the contribution of each code component to the overall execution time.

## 📊 Profiling Results

### **Run 0 (Top-P pipeline):**
End-to-end profiling of the top-p sampling CUDA pipeline (softmax, sort, nucleus, sample).

> Outcome: Radix Sort step highlighted as the primary contributor to overall latency, especially at large vocabulary sizes.

See: [top_p/README.md](top_p/README.md) for the full breakdown and charts.

### **Run 1 (Launcher profiling):**
Measured timings with Nvprof and cudaEventRecord to profile launcher overheads.

> Outcome: Found host-side looping dominates for small vocab sizes while GPU radix kernels dominate at 1M tokens; documents a large Nvprof vs cudaEvent timing discrepancy and recommends moving per-bit offset computation onto the GPU to eliminate host-device synchronization overheads.

See: [prof/results/run1.md](prof/results/run1.md).

### **Run 2 (Nsight Compute profiling):**
Isolated profiling of `prefix_per_block` and `radix_sort`, and experimented with moving inputs into shared memory.

> Outcome: Shared-memory changes reduced long-scoreboard stalls but increased other stalls and slowed the `prefix` kernel (~20%). The dominant remaining issue is uncoalesced/sparse global stores (per-block scatter to `block_sums` and output writes); recommended next steps are computing offsets on‑GPU or redesigning block partitioning to eliminate sparse writes.

See: [prof/results/run2.md](prof/results/run2.md).

### **Run 3 (Hillis-Steele vs on-host loop):**
Replaced the host-side per-batch prefix loop with a GPU Hillis–Steele exclusive-sum kernel.

> Outcome: Removes host–device synchronization and introduces per-batch parallelism (Hillis–Steele contributes only a tiny fraction of per-bit loop latency), yielding the largest end-to-end speedups at smaller vocabularies (over `3x`). `prefix_per_block` and `radix` remain the dominant costs for large inputs.

See: [prof/results/run3.md](prof/results/run3.md).

---

## 🚀 Setup

### **Compile**
```bash
make clean && make NVCC_ARCH=XX  
```

### **Smoke-run**
```bash
./bin/runner --kernel=radix_v1 --vocab_size=32768 --num_batches=1 --warmup_runs=1 --runs=1
```

### **Quick Profile Run**
```bash
ncu ./bin/profile_harness --num_batches=256 --vocab_size=32768 --kernel=radix_v1 --warmup_runs=0 --runs=1 > prof/txt/radix_v1_256_32768.txt
```

### **Nvprof Profile**
```bash
nvprof ./bin/profile_harness --num_batches=256 --vocab_size=32768 --kernel=radix_v1 --warmup_runs=0 --runs=1
```

or for per-call detail:
```bash
nvprof --print-gpu-trace ./bin/profile_harness --num_batches=256 --vocab_size=32768 --kernel=radix_v1 --warmup_runs=0 --runs=1 2>&1 | tee prof/txt/nvprof_gputrace_radix_v1_256_32768.txt
```

### **Nsight Compute Profile**
```bash
ncu --import-source yes --set full --export prof/ncu/radix_v1.ncu-rep ./bin/profile_harness --kernel=radix_v1 --warmup_runs=1 --runs=2
```

### **Test**
```bash
./bin/test_sort --vocab_size 1048576 --num_batches 64 --verbose
```

---