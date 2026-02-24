# RadixSortNCU

## Overview

This project profiles and optimizes the **Radix Sort** kernel pipeline for GPU-accelerated top-p sampling. 

The analysis reveals that sorting becomes the primary bottleneck in large-vocabulary pipelines, accounting for up to **86%** of latency at vocabulary sizes of 1M tokens. Through systematic profiling of the complete radix sort launcher—including both GPU kernels and CPU-side orchestration—we identify and measure the contribution of each code component to the overall execution time. 
---

## 📊 Profiling Results

### **Run 0 (Top-P pipeline):**
End-to-end profiling of the top-p sampling CUDA pipeline (softmax, sort, nucleus, sample) highlights the Radix Sort step as the primary contributor to overall latency at large vocabulary sizes.
See: [top_p/README.md](top_p/README.md) for the full breakdown and charts.

### **Run 1 (Launcher profiling):**
Measured both Nvprof and cudaEventRecord timings; found host-side looping dominates for small vocab sizes while GPU radix kernels dominate at 1M tokens. See the full report for tables and charts. It also documents a large Nvprof vs cudaEvent timing discrepancy and recommends moving per-bit offset computation onto the GPU to eliminate host-device synchronization overheads.
See: [prof/results/run1.md](prof/results/run1.md).

### **Run 2 (Nsight Compute profiling):**
Isolated profiling `prefix_per_block` and `radix_sort`. Moving inputs into shared memory reduced long-scoreboard stalls in `radix_v2` but increased other stalls and caused the `prefix` kernel to slow (~20%). The dominant remaining issue is uncoalesced/sparse global stores (per-block scatter to `block_sums` and output writes); recommended next steps are computing offsets on‑GPU or redesigning the block partitioning to eliminate sparse writes.
See: [prof/results/run2.md](prof/results/run2.md).

### **Run 3 (Replacing on-host loop with Hillis-Steele kernel):**
Replaced the host-side per-batch prefix loop with a GPU Hillis–Steele exclusive-sum kernel. This change removes host–device synchronization and introduces per-batch parallelism (the Hillis–Steele kernel contributes only a tiny fraction of iter-loop latency), yielding the largest end-to-end speedups at smaller vocabularies (over 3x) while `prefix_per_block` and `radix` remain the dominant costs for large inputs. See: [prof/results/run3.md](prof/results/run3.md).
---

## 🚀 Set-up

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