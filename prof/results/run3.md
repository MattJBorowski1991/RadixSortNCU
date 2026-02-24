# Run 3 — Hillis–Steele GPU exclusive-sum replacement (radix_v3)

Summary
- Replaced the on-host per-batch prefix loop with a GPU Hillis–Steele exclusive-sum kernel (`hillis_steele_prefix_excl`).
- Results below show full launcher latency, the latency of the inner 32-bit iteration loop, and per-kernel timing proportions inside the iter-loop.
- Outcome: the Hillis–Steele kernel adds negligible cost (~16 µs) while removing host/device syncs; most time remains in `prefix_per_block` and `radix` kernels.
- Unsuprisingly the largest benefit is observed for the smallest vocab size where the on-host loop was the largest bottleneck (see [Run 1](https://github.com/MattJBorowski1991/RadixSortNCU/blob/master/prof/results/run1.md)).

## I. Full launcher latency (ms)
| Vocab Size | Before (radix_v1) | After (radix_v3) | Speedup |
|---:|---:|---:|---:|
| 32,768  | 99.1  | 26.66   | 3.72x |
| 131,072 | 185.56  | 104.58  | 1.78x |
| 524,288 | 522.90 | 416.80 | 1.25x |
| 1,048,576 | 969.5 | 832.27 | 1.17x |

## II. Latency breakdown (ms)

<table>
	<thead>
		<tr>
			<th rowspan="2">Vocab Size</th>
			<th colspan="4">Latency (ms)</th>
			<th rowspan="2">mem transfers &amp; init</th>
		</tr>
		<tr>
			<th>Total</th>
			<th>prefix_per_block</th>
			<th>hillis_steele</th>
			<th>radix</th>
		</tr>
	</thead>
	<tbody>
		<tr><td>32,768</td><td>26.66</td><td>29.7%</td><td>3.6%</td><td>46.8%</td><td>1.704</td></tr>
		<tr><td>131,072</td><td>104.58</td><td>32.1%</td><td>1.0%</td><td>51.1%</td><td>4.897</td></tr>
		<tr><td>524,288</td><td>416.80</td><td>34.8%</td><td>0.3%</td><td>55.9%</td><td>10.369</td></tr>
		<tr><td>1,048,576</td><td>832.27</td><td>35.6%</td><td>0.2%</td><td>56.9%</td><td>16.561</td></tr>
	</tbody>
</table>

**Notes**

- **Hillis–Steele cost.** The GPU Hillis–Steele exclusive-sum kernel is very small in absolute time (~0.3–0.35 ms measured here) and does not increase significantly with vocabulary size; it is not on the critical path for large inputs.

- **Dominant work.** The `prefix_per_block` and `radix` kernels account for the vast majority of the iter-loop latency and scale roughly linearly with input size. Further performance work should prioritise those kernels.

- **Cooperative launch limitation.** A single cooperative kernel launch requires all resident blocks used by the launch to fit within the device's cooperative capacity (SMs × maxBlocksPerSM). On a T4 (40 SMs × 16 blocks/SM = 640 blocks) a full-grid launch for `vocab_size=1,048,576` with `num_batches=256` would need 65,536 blocks, which exceeds the device limit; hence the cooperative approach is not feasible for this workload. See `kernels/radix_v3_coop.cu` for the attempted implementation and failure notes.

- **Next steps.** Verify correctness across inputs, run Nsight Compute to inspect L1/cache behavior and kernel stalls, and focus optimization effort on `prefix_per_block` and `radix` where the largest wins are likely.