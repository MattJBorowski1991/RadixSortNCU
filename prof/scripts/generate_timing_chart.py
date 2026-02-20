import os
import matplotlib.pyplot as plt

# Data from README (cudaEvent Timings Summary)
vocab_labels = ['32k', '130k', '520k', '1050k']

# Components as percent of total latency (from cudaEvent Timings Summary)
# mem_init = DtoD buffer + HtoD total_ones + DtoD output + init_indices
mem_init = [4.2, 5.1, 4.7, 4.5]
# prefix kernel percent
prefix = [7.7, 15.6, 21.8, 23.6]
# on-host loop percent
host_loop = [68.9, 38.2, 15.2, 9.3]
# radix kernel percent
radix = [19.2, 41.1, 58.3, 62.6]

# Absolute latency (ms) used for group annotations
latency_ms = [49, 92, 261, 484]

# Grouped bar chart
x = range(len(vocab_labels))
width = 0.18

fig, ax = plt.subplots(figsize=(10,6))
ax.bar([p - 1.5*width for p in x], mem_init, width=width, label='mem transfers & init', color='#4C72B0')
ax.bar([p - 0.5*width for p in x], prefix, width=width, label='prefix kernel', color='#55A868')
ax.bar([p + 0.5*width for p in x], host_loop, width=width, label='on-host loop', color='#C44E52')
ax.bar([p + 1.5*width for p in x], radix, width=width, label='radix kernel', color='#8172B2')

ax.set_xticks(x)
ax.set_xticklabels(vocab_labels)
ax.set_xlabel('Vocab Size')
ax.set_ylabel('% of total latency')
ax.set_title('Radix sort - cudaEventRecord timings')
ax.set_ylim(0, 100)
ax.legend()

for i in x:
    ax.text(i - 1.5*width, mem_init[i] + 0.8, f"{mem_init[i]:.1f}%", ha='center', va='bottom', fontsize=8)
    ax.text(i - 0.5*width, prefix[i] + 0.8, f"{prefix[i]:.1f}%", ha='center', va='bottom', fontsize=8)
    ax.text(i + 0.5*width, host_loop[i] + 0.8, f"{host_loop[i]:.1f}%", ha='center', va='bottom', fontsize=8)
    ax.text(i + 1.5*width, radix[i] + 0.8, f"{radix[i]:.1f}%", ha='center', va='bottom', fontsize=8)
    # Group annotation: Latency = X ms
    ax.text(i, max(mem_init[i], prefix[i], host_loop[i], radix[i]) + 6, f"Latency = {latency_ms[i]} ms", ha='center', va='bottom', fontsize=9, fontweight='bold')

out_dir = os.path.join('prof', 'images', 'run1')
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, 'radix_event_timing_chart.png')
plt.tight_layout()
plt.savefig(out_path, dpi=150)
print('Wrote', out_path)
