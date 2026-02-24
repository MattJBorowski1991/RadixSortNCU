import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Data from prof/results/run3.md (latency breakdown)
labels = ['32k', '130k', '520k', '1050k']

# Absolute component latencies (ms)
prefix = np.array([10.04, 39.78, 159.50, 318.82])
hillis = np.array([0.96, 0.96, 1.01, 1.07])
radix = np.array([16.00, 63.92, 256.23, 511.07])
mem = np.array([5.10, 10.89, 26.68, 46.58])

# Percent contributions (from correct_table.txt) - used for Y axis bars
prefix_pct = np.array([31.3, 34.4, 36.0, 36.3])
hillis_pct = np.array([3.0, 0.8, 0.2, 0.1])
radix_pct = np.array([49.9, 55.3, 57.8, 58.2])
mem_pct = np.array([15.9, 9.4, 6.0, 5.3])

# Total latencies for group annotations
latency_ms = np.array([32.10, 115.55, 443.41, 877.54])

# Full-launch latencies (before) and computed speedups
before_ms = np.array([99.1, 185.56, 522.90, 969.5])
speedups = before_ms / latency_ms
speedups_rounded = np.round(speedups, 1)

x = np.arange(len(labels))
n_groups = 4
bar_width = 0.18

fig, ax = plt.subplots(figsize=(9,5))

# Plot percent values; order: mem (leftmost), prefix, hillis, radix
ax.bar([p - 1.5*bar_width for p in x], mem_pct, bar_width, label='mem transfers & init', color='#4C72B0')
ax.bar([p - 0.5*bar_width for p in x], prefix_pct, bar_width, label='prefix_per_block', color='#55A868')
ax.bar([p + 0.5*bar_width for p in x], hillis_pct, bar_width, label='hillis_steele', color='#C44E52')
ax.bar([p + 1.5*bar_width for p in x], radix_pct, bar_width, label='radix', color='#8172B2')

# Labels and styling
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_xlabel('Vocab Size')
# Y axis: percent, cap at 100
# (remove explicit y-axis label per request)
ax.set_ylim(0, 100)
ax.set_title('Latency breakdown per vocab size')
ax.legend(loc='upper right', frameon=True, fontsize=8)

# Annotate each bar with its percent contribution
for i in range(len(x)):
    ax.text(x[i] - 1.5*bar_width, mem_pct[i] + 1.5, f"{mem_pct[i]:.1f}%", ha='center', va='bottom', fontsize=9)
    ax.text(x[i] - 0.5*bar_width, prefix_pct[i] + 1.5, f"{prefix_pct[i]:.1f}%", ha='center', va='bottom', fontsize=9)
    ax.text(x[i] + 0.5*bar_width, hillis_pct[i] + 1.5, f"{hillis_pct[i]:.1f}%", ha='center', va='bottom', fontsize=9)
    ax.text(x[i] + 1.5*bar_width, radix_pct[i] + 1.5, f"{radix_pct[i]:.1f}%", ha='center', va='bottom', fontsize=9)

# Compute a single uniform annotation level (based on the largest group's top)
global_top = max(np.max(mem_pct), np.max(prefix_pct), np.max(hillis_pct), np.max(radix_pct))
latency_y = global_top + 14.0
speedup_y = global_top + 8.0
for i in range(len(x)):
    ax.text(x[i], latency_y, f"Latency = {latency_ms[i]:.0f} ms", ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax.text(x[i], speedup_y, f"Speedup* {speedups_rounded[i]:.1f}x", ha='center', va='bottom', fontsize=9)

# Add explanatory footnote text below the plot (centered and moved lower)
plt.subplots_adjust(bottom=0.40)
# place the footnote lower
fig.text(0.5, 0.002, "*speedup vs pipeline with on-host per-batch per-block loop i.e. radix_v1", ha='center', fontsize=8)

plt.tight_layout()
out_path = '/teamspace/studios/this_studio/RadixSortNCU/prof/images/run3/radix_event_timing_chart.png'
plt.savefig(out_path, dpi=200)
print('Saved', out_path)
