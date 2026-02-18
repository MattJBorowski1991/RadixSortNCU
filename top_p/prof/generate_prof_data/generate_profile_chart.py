#!/usr/bin/env python3
"""
Professional profiling results visualization script.
Generates a stacked column chart showing ns/tok breakdown by percentage contributions.
"""

import re
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches

# Professional color palette (warm tones for visual appeal)
COLORS = [
    '#2E86AB',  # Deep Blue - STEP 1: softmax
    '#A23B72',  # Deep Magenta - STEP 2.1: sort
    '#F18F01',  # Vibrant Orange - STEP 2.2: reverse
    '#C73E1D',  # Deep Red - STEP 3.1: nucleus
    '#6A994E',  # Sage Green - STEP 3.2: norm+pref
    '#10B981',  # Muted Teal - STEP 4: sample (softer, less aggressive than purple)
]

STEP_ORDER = [
    'STEP 1: softmax',
    'STEP 2.1: sort',
    'STEP 2.2: reverse',
    'STEP 3.1: nucleus',
    'STEP 3.2: norm+pref',
    'STEP 4: sample',
]

# Clean labels for legend (remove STEP prefixes)
LEGEND_LABELS = [
    'softmax',
    'sort',
    'reverse',
    'nucleus',
    'norm+pref',
    'sample',
]

def parse_profiling_results(filepath):
    """Parse the profiling results file and extract relevant data."""
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Initialize data storage
    data = {}
    
    # Split by vocab_size entries
    vocab_blocks = re.split(r'vocab_size = (\d+)', content)
    
    # Process each vocab_size block
    for i in range(1, len(vocab_blocks), 2):
        if i + 1 >= len(vocab_blocks):
            break
            
        vocab_size = int(vocab_blocks[i])
        block_content = vocab_blocks[i + 1]
        
        # Extract TIMING RESULTS section
        timing_match = re.search(r'========== TIMING RESULTS =+\n(.*?)\n=+', block_content, re.DOTALL)
        if not timing_match:
            continue
        
        timing_section = timing_match.group(1)
        lines = timing_section.strip().split('\n')
        
        # Find the data rows (skip header and separator lines)
        steps_data = {}
        total_ns_tok = None
        
        for line in lines:
            # Skip header, separator lines, and empty lines
            if not line.strip() or '|' not in line or line.strip().startswith('-') or 'Name' in line:
                continue
            
            parts = [p.strip() for p in line.split('|')]
            
            # Handle TOTAL line separately (look for it by presence of "TOTAL")
            if parts[0] == 'TOTAL' or 'TOTAL' in parts[0]:
                try:
                    # TOTAL line format: TOTAL | Time | % | ns/tok
                    # parts: [0]=Name, [1]=Time(ms), [2]=%, [3]=ns/tok
                    total_ns_tok = float(parts[3])
                except (ValueError, IndexError):
                    pass
                continue
            
            # Skip H2D Transfer line
            if 'H2D Transfer' in parts[0]:
                continue
            
            # Skip empty or incomplete lines
            if len(parts) < 4 or not parts[0].startswith('STEP'):
                continue
            
            step_name = parts[0].strip()
            
            # Extract percentage (parts[2] = %)
            try:
                percentage_str = parts[2].replace('%', '').strip()
                percentage = float(percentage_str)
            except (ValueError, IndexError):
                continue
            
            # Extract ns/tok value (parts[3])
            try:
                ns_tok = float(parts[3])
            except (ValueError, IndexError):
                continue
            
            steps_data[step_name] = {
                'percentage': percentage,
                'ns_tok': ns_tok
            }
        
        if steps_data and total_ns_tok is not None:
            data[vocab_size] = {
                'steps': steps_data,
                'total_ns_tok': total_ns_tok
            }
    
    return data

def create_visualization(data, output_path='profiling_chart.png'):
    """Create a professional stacked column chart visualization."""
    
    # Sort vocab sizes for x-axis
    vocab_sizes = sorted(data.keys())
    vocab_labels = [f'{v//1024}K' if v >= 1024 else str(v) for v in vocab_sizes]
    
    # Prepare data for stacked bar chart
    step_percentages = {step: [] for step in STEP_ORDER}
    ns_tok_values = []
    
    for vocab_size in vocab_sizes:
        ns_tok_values.append(data[vocab_size]['total_ns_tok'])
        
        for step in STEP_ORDER:
            if step in data[vocab_size]['steps']:
                step_percentages[step].append(data[vocab_size]['steps'][step]['percentage'])
            else:
                step_percentages[step].append(0.0)
    
    # Create figure with professional styling
    fig, ax = plt.subplots(figsize=(16, 9), dpi=300)
    
    # Set background color to white for professional look
    fig.patch.set_facecolor('white')
    ax.set_facecolor('#fafbfc')
    
    # Create stacked bars
    x_pos = np.arange(len(vocab_sizes))
    width = 0.65
    
    bottom = np.zeros(len(vocab_sizes))
    bars = []
    
    for idx, step in enumerate(STEP_ORDER):
        bars.append(ax.bar(x_pos, step_percentages[step], width,
                          label=LEGEND_LABELS[idx], bottom=bottom, color=COLORS[idx],
                          edgecolor='white', linewidth=2.0))
        bottom += np.array(step_percentages[step])
    
    # Add percentage labels on segments (enhanced)
    for idx, step in enumerate(STEP_ORDER):
        for i, (bar_container, percent) in enumerate(zip(bars[idx], step_percentages[step])):
            if percent > 2:  # Only show label if segment is large enough
                height = bar_container.get_height()
                y_pos = bar_container.get_y() + height / 2
                
                # Create text with better visibility
                ax.text(bar_container.get_x() + width/2, y_pos, f'{percent:.1f}%',
                       ha='center', va='center', fontsize=10, fontweight='bold',
                       color='white',
                       bbox=dict(boxstyle='round,pad=0.4',
                                facecolor='black', alpha=0.4, edgecolor='white', linewidth=0.5))
    
    # Add ns/tok value labels above bars with better formatting
    for i, (x, ns_tok) in enumerate(zip(x_pos, ns_tok_values)):
        # Add a box with the ns/tok value - positioned right above the bars
        ax.text(x, 102, f'{ns_tok:.0f} ns/tok', ha='center', va='bottom',
               fontsize=11, fontweight='normal', color='#1a1a1a')
    
    # Customize axes
    ax.set_xlabel('Vocabulary Size', fontsize=14, fontweight='bold', labelpad=15, color='#1a1a1a')
    
    # Add footnote in very small text below the x-axis label
    ax.text(0.5, -0.15, '* Separate CUDA kernels were profiled and timed for each of the operations with batch size = 64, p=0.95, and inputs that resulted in nucleus of sizes from 100 to 500',
            ha='center', va='top', fontsize=12, style='italic', color='#666666',
            transform=ax.transAxes, linespacing=1.2)
    ax.set_ylabel('Percentage Contribution (%)', fontsize=14, fontweight='bold', labelpad=15, color='#1a1a1a')
    
    # Title with subtitle
    ax.text(0.5, 1.08, 'Inference Profiling Results: Last step of Decode*',
            ha='center', va='bottom', fontsize=17, fontweight='bold',
            transform=ax.transAxes, color='#1a1a1a')
    ax.text(0.5, 1.02, 'ns/tok Performance by Vocabulary Size - Radix Sort Nucleus Sampling Pipeline',
            ha='center', va='bottom', fontsize=11, style='italic',
            transform=ax.transAxes, color='#666666')
    
    # Set x-axis
    ax.set_xticks(x_pos)
    ax.set_xticklabels(vocab_labels, fontsize=12, fontweight='bold', color='#1a1a1a')
    
    # Set y-axis limits and ticks
    ax.set_ylim(0, 110)
    ax.set_yticks(np.arange(0, 101, 20))
    ax.set_yticklabels([f'{int(y)}%' for y in np.arange(0, 101, 20)], fontsize=11, color='#555555')
    
    # Add grid for better readability
    ax.grid(axis='y', linestyle='--', alpha=0.25, linewidth=0.9, color='#888888')
    ax.set_axisbelow(True)
    
    # Improve spines - cleaner look
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_color('#444444')
    ax.spines['bottom'].set_color('#444444')
    
    # Improve tick parameters
    ax.tick_params(axis='both', which='major', labelsize=11, colors='#555555', width=1.5, length=6)
    
    # Create legend with professional styling - positioned outside
    legend = ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1), frameon=True, fontsize=11,
                      framealpha=0.98, edgecolor='#cccccc', fancybox=True, shadow=True,
                      title='Kernels', title_fontsize=12)
    legend.get_frame().set_linewidth(1.5)
    legend.get_title().set_fontweight('bold')
    legend.get_title().set_color('#1a1a1a')
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    # Save with high quality
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"✓ Chart saved to: {output_path}")
    
    return fig, ax

def print_data_summary(data):
    """Print a summary of the parsed data."""
    print("\n" + "="*80)
    print("PROFILING DATA SUMMARY - Radix Sort Nucleus Sampling Pipeline")
    print("="*80)
    
    for vocab_size in sorted(data.keys()):
        print(f"\n📊 Vocabulary Size: {vocab_size:>7} | Total ns/tok: {data[vocab_size]['total_ns_tok']:>10.2f}")
        print("  " + "-"*76)
        print("  Operation              │  Contribution  │   ns/tok   │  % of Total")
        print("  " + "-"*76)
        for step in STEP_ORDER:
            if step in data[vocab_size]['steps']:
                info = data[vocab_size]['steps'][step]
                pct = info['percentage']
                ns_tok = info['ns_tok']
                # Simplified step name for display
                short_name = step.replace('STEP ', '').replace(': ', ': ')[:22]
                print(f"  {short_name:22} │  {pct:>6.2f}%      │  {ns_tok:>8.2f}  │  {(pct/100)*100:>5.1f}%")
    print("\n" + "="*80)

if __name__ == '__main__':
    import sys
    
    # Get input file path
    input_file = '/teamspace/studios/this_studio/RadixSortNCU/prof/prof_results/prof_results_15022026.txt'
    output_file = '/teamspace/studios/this_studio/RadixSortNCU/prof/prof_results/profiling_visualization.png'
    
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    if len(sys.argv) > 2:
        output_file = sys.argv[2]
    
    print(f"Parsing profiling results from: {input_file}")
    
    # Parse data
    data = parse_profiling_results(input_file)
    
    if not data:
        print("Error: Could not parse any profiling data!")
        sys.exit(1)
    
    # Print summary
    print_data_summary(data)
    
    # Create visualization
    print(f"\nGenerating visualization...")
    create_visualization(data, output_file)
    
    print("\nVisualization complete!")
