"""
Generate radar plot showing path error rates averaged over all model sizes.
Compares Baseline, SFT, and GRPO configurations.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from pathlib import Path
from plot_config import setup_plot_style, COLUMN_WIDTH_INCHES, get_training_method_color

# Setup plot style
setup_plot_style(use_latex=True)


def read_stats_file(filepath):
    """Read a stats CSV file and extract path validation metrics."""
    df = pd.read_csv(filepath)
    
    # Find the path validation metrics
    metrics = {}
    for _, row in df.iterrows():
        metric_name = row['Metric']
        if metric_name in ['Fully Valid Paths', 'Connected Paths', 'Correct Start/End', 
                          'Non-Intersecting', 'No Rule Violations']:
            # Extract percentage value
            percentage_str = row['Percentage'].strip('%')
            metrics[metric_name] = float(percentage_str)
    
    return metrics


def calculate_error_rates(success_metrics):
    """Convert success rates to error rates (100 - success_rate)."""
    metric_mapping = {
        'Fully Valid Paths': 'Invalid\nPath',
        'Correct Start/End': 'Incorrect\nStart/End',
        'Connected Paths': 'Disconnected\nLine',
        'Non-Intersecting': 'Intersecting\nLine',
        'No Rule Violations': 'Rule Cell\nCrossing'
    }
    
    metrics = {}
    for success_metric, display_label in metric_mapping.items():
        if success_metric in success_metrics:
            metrics[display_label] = 100.0 - success_metrics[success_metric]
        else:
            metrics[display_label] = 100.0  # If missing, assume 100% error
    
    return metrics


def generate_averaged_config_plot(results_dir, output_dir):
    """
    Generate a radar plot averaging over all model sizes to show configuration differences.
    Compares Baseline, SFT, and GRPO configurations showing error rates.
    """
    results_dir = Path(results_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model_sizes = ['0.6B', '4B', '14B', '32B']
    
    # Define configurations - only showing Baseline, SFT, and GRPO for clarity
    configs_base = {
        'Baseline': 'Qwen_Qwen3-{size}_stats.csv',
        'SFT': 'lkaesberg_Qwen3-{size}-SPaRC-SFT_stats.csv',
        'GRPO': 'lkaesberg_Qwen3-{size}-SPaRC-GRPO_stats.csv',
    }
    
    # Define colors for configurations using centralized color scheme
    config_colors = {
        'Baseline': get_training_method_color('Baseline'),
        'SFT': get_training_method_color('SFT'),
        'GRPO': get_training_method_color('GRPO'),
        'GRPO-8E': get_training_method_color('GRPO'),  # Same as GRPO
        'GRPO-16R': get_training_method_color('GRPO'), # Same as GRPO
        'GRPO-L': get_training_method_color('GRPO-L'),
    }
    
    config_markers = {
        'Baseline': 'X',     # X marker
        'SFT': 'o',          # Circle
        'GRPO': 's',         # Square
        'GRPO-8E': '^',      # Triangle up
        'GRPO-16R': 'D',     # Diamond
        'GRPO-L': 'P',       # Plus (filled)
    }
    
    # Collect data for each configuration, averaged across model sizes
    config_averaged_data = {}
    
    # Define error metric types
    metric_types = [
        'Invalid\nPath',
        'Incorrect\nStart/End',
        'Disconnected\nLine',
        'Intersecting\nLine',
        'Rule Cell\nCrossing'
    ]
    
    for config_name, filename_template in configs_base.items():
        all_metrics = []
        
        for size in model_sizes:
            filename = filename_template.format(size=size)
            filepath = results_dir / filename
            
            if filepath.exists():
                success_metrics = read_stats_file(filepath)
                metrics = calculate_error_rates(success_metrics)
                
                metric_rates = [metrics.get(metric_type, 100.0) for metric_type in metric_types]
                all_metrics.append(metric_rates)
        
        if all_metrics:
            # Average across all model sizes
            config_averaged_data[config_name] = np.mean(all_metrics, axis=0)
    
    if not config_averaged_data:
        print("No data found for averaged configuration plot")
        return
    
    fig = plt.figure(figsize=(COLUMN_WIDTH_INCHES, COLUMN_WIDTH_INCHES * 2))
    ax = fig.add_subplot(111, polar=True)
    ax.set_facecolor('white')
    
    N = len(metric_types)
    theta = np.linspace(0, 2*np.pi, N, endpoint=False)
    theta = np.append(theta, theta[0])
    
    ax.set_ylim(0, 62)
    
    # Plot order for legend clarity
    plot_order = ['Baseline', 'SFT', 'GRPO']
    
    for config_name in plot_order:
        if config_name not in config_averaged_data:
            continue
        
        metric_rates_closed = np.append(config_averaged_data[config_name], 
                                       config_averaged_data[config_name][0])
        
        color = config_colors[config_name]
        marker = config_markers[config_name]
        
        ax.plot(theta, metric_rates_closed, marker=marker, linestyle='-', 
               linewidth=1.5, color=color, 
               label=config_name, alpha=0.9, markersize=4,
               markerfacecolor=color, markeredgecolor='white', markeredgewidth=1.0,
               zorder=1)
        
        ax.fill(theta, metric_rates_closed, alpha=0.1, color=color, zorder=1)
    
    ax.set_xticks(theta[:-1])
    ax.set_xticklabels(metric_types)
    ax.tick_params(axis='x', pad=8)
    
    ax.set_yticks([20, 40, 60])
    ax.set_yticklabels(['20\\%', '40\\%', '60\\%'])
    
    ax.yaxis.grid(True, color='#888888', linestyle='-', linewidth=0.5, alpha=0.3)
    ax.xaxis.grid(True, color='#888888', linestyle='-', linewidth=0.5, alpha=0.3)
    ax.spines['polar'].set_visible(False)
    
    # Set y-axis labels to appear in front of plot lines with white outline
    ax.tick_params(axis='y', which='major', pad=-20, labelsize=9)
    for label in ax.yaxis.get_ticklabels():
        label.set_zorder(100)
        label.set_path_effects([
            path_effects.Stroke(linewidth=2, foreground='white'),
            path_effects.Normal()
        ])
        
    legend = ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.18), 
                     ncol=3, frameon=False, facecolor='white',
                     edgecolor='#dddddd', framealpha=1.0,
                     handletextpad=0.5, columnspacing=1.0)
    
    plt.tight_layout()
    
    output_base = output_dir / 'radar_path_errors_averaged_configs'
    print(config_averaged_data)
    print("Labels: ", metric_types)
    plt.savefig(f'{output_base}.pdf', dpi=300, bbox_inches='tight', format='pdf')
    plt.savefig(f'{output_base}.png', dpi=300, bbox_inches='tight')
    print(f"Averaged configuration plot saved: {output_base}.pdf and {output_base}.png")
    
    plt.close()


if __name__ == '__main__':
    results_dir = Path(__file__).parent / 'results' / 'sparc'
    output_dir = Path(__file__).parent / 'results' / 'figures'
    
    print(f"\n{'='*60}")
    print(f"Generating averaged configuration radar plot (error rates)")
    print(f"{'='*60}")
    
    generate_averaged_config_plot(results_dir, output_dir)
    
    print(f"\n{'='*60}")
    print(f"Radar plot generated successfully!")
    print(f"{'='*60}\n")
