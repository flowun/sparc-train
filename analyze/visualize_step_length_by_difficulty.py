#!/usr/bin/env python3
"""
Visualize average step length by difficulty level across model variants.
Compares Baseline Qwen, SFT, GRPO-L, and Step-by-step approaches.
Averages across all model sizes (0.6B, 4B, 14B, 32B) for each difficulty level.
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import re
import json

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))
from plot_config import setup_plot_style, TEXT_WIDTH_INCHES, COLUMN_WIDTH_INCHES, get_training_method_color

def parse_step_by_step_difficulty_file(filepath):
    """Parse step-by-step summary_by_difficulty file to extract steps per difficulty."""
    with open(filepath, 'r') as f:
        content = f.read()
    
    difficulty_data = {}
    # Match patterns like "Difficulty 1" followed by "steps_per_puzzle: avg=9.60"
    pattern = r'Difficulty (\d+).*?steps_per_puzzle: avg=([\d.]+)'
    matches = re.findall(pattern, content, re.DOTALL)
    
    for difficulty, avg_steps in matches:
        difficulty_data[int(difficulty)] = float(avg_steps)
    
    return difficulty_data

def get_optimal_solution_stats(filepath):
    """
    Extract optimal path lengths from a JSONL file.
    
    Args:
        filepath: Path to a JSONL file containing puzzle definitions
        
    Returns:
        Dict mapping difficulty level to average optimal path length
    """
    difficulty_lengths = {1: [], 2: [], 3: [], 4: [], 5: []}
    
    with open(filepath, 'r') as f:
        for line in f:
            try:
                data = json.loads(line)
                difficulty = data.get('difficulty_level')
                solutions = data.get('solutions', [])
                
                if difficulty in difficulty_lengths and solutions:
                    # Use the length of the first solution as the optimal length
                    # (Usually the provided solutions are optimal or reference)
                    path_length = solutions[0].get('pathLength')
                    if path_length is not None:
                        difficulty_lengths[difficulty].append(path_length)
            except json.JSONDecodeError:
                continue
                
    # Calculate averages
    avg_optimal_lengths = {}
    for diff, lengths in difficulty_lengths.items():
        if lengths:
            avg_optimal_lengths[diff] = np.mean(lengths)
        else:
            avg_optimal_lengths[diff] = None
            
    return avg_optimal_lengths

def calculate_average_by_difficulty(results_dir, model_sizes, file_pattern):
    """
    Calculate average path length by difficulty across multiple model sizes.
    
    Args:
        results_dir: Path to results directory
        model_sizes: List of model size strings (e.g., ['0.6B', '4B', '14B', '32B'])
        file_pattern: Pattern for filename with {} placeholder for size
    
    Returns:
        Dict mapping difficulty level to average path length
    """
    all_data = []
    
    for size in model_sizes:
        filename = file_pattern.format(size=size)
        filepath = results_dir / filename
        
        if not filepath.exists():
            print(f"Warning: {filepath} not found, skipping...")
            continue
        
        # Read CSV file
        df = pd.read_csv(filepath)
        all_data.append(df)
    
    # Concatenate all dataframes
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Calculate average path length by difficulty
    avg_by_difficulty = combined_df.groupby('Difficulty')['Path Length'].mean()
    
    return avg_by_difficulty.to_dict()

def create_step_length_by_difficulty_chart():
    """Create line chart comparing average step lengths by difficulty level."""
    
    # Setup plot style
    setup_plot_style(use_latex=True)
    
    results_dir = Path(__file__).parent / "results" / "sparc"
    step_by_step_dir = Path(__file__).parent / "results" / "step-by-step"
    
    model_sizes = ['0.6B', '4B', '14B', '32B']
    
    # Calculate averages for each variant
    print("Calculating averages across model sizes...")
    
    # Baseline Qwen
    baseline_data = calculate_average_by_difficulty(
        results_dir, model_sizes, 
        'Qwen_Qwen3-{size}_details.csv'
    )
    
    # SFT
    sft_data = calculate_average_by_difficulty(
        results_dir, model_sizes,
        'lkaesberg_Qwen3-{size}-SPaRC-SFT_details.csv'
    )
    
    # GRPO-L
    grpo_l_data = calculate_average_by_difficulty(
        results_dir, model_sizes,
        'lkaesberg_Qwen3-{size}-SPaRC-GRPO-L_details.csv'
    )
    
    # Step-by-step - average across model sizes
    step_by_step_by_difficulty = {1: [], 2: [], 3: [], 4: [], 5: []}
    for size in model_sizes:
        filepath = step_by_step_dir / f'summary_by_difficulty_Qwen3-{size}.txt'
        if filepath.exists():
            data = parse_step_by_step_difficulty_file(filepath)
            for diff, steps in data.items():
                step_by_step_by_difficulty[diff].append(steps)
    
    # Average step-by-step data
    step_by_step_data = {
        diff: np.mean(values) if values else None 
        for diff, values in step_by_step_by_difficulty.items()
    }
    
    # Real Solution (Optimal)
    # Use one of the JSONL files to get the ground truth path lengths
    # (Puzzle definitions are the same across all result files)
    optimal_data_source = results_dir / 'Qwen_Qwen3-0.6B.jsonl'
    if optimal_data_source.exists():
        print(f"Calculating optimal solution lengths from {optimal_data_source.name}...")
        optimal_data = get_optimal_solution_stats(optimal_data_source)
    else:
        print(f"Warning: Could not find {optimal_data_source} for optimal solution stats")
        optimal_data = {}
    
    # Prepare data for plotting
    difficulties = [1, 2, 3, 4, 5]
    
    baseline_values = [baseline_data.get(d, np.nan) for d in difficulties]
    sft_values = [sft_data.get(d, np.nan) for d in difficulties]
    grpo_l_values = [grpo_l_data.get(d, np.nan) for d in difficulties]
    step_by_step_values = [step_by_step_data.get(d, np.nan) for d in difficulties]
    optimal_values = [optimal_data.get(d, np.nan) for d in difficulties]
    
    # Print summary
    print("\n" + "="*80)
    print("AVERAGE PATH LENGTH BY DIFFICULTY (averaged across all model sizes)")
    print("="*80)
    print(f"\n{'Difficulty':<12} {'Baseline':>12} {'SFT':>12} {'GRPO-L':>12} {'Step-by-step':>15} {'Real Sol.':>12}")
    print("-" * 92)
    
    for i, diff in enumerate(difficulties):
        print(f"{diff:<12} {baseline_values[i]:>12.2f} {sft_values[i]:>12.2f} "
              f"{grpo_l_values[i]:>12.2f} {step_by_step_values[i]:>15.2f} {optimal_values[i]:>12.2f}")
    
    # Create figure
    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH_INCHES, COLUMN_WIDTH_INCHES))
    
    # Bar chart configuration
    x = np.arange(len(difficulties))  # Label locations
    width = 0.15  # Width of each bar (made smaller to fit 5 bars)
    
    # Create grouped bars with consistent colors from plot_config
    bars1 = ax.bar(x - 2.0*width, baseline_values, width, 
                   label='Baseline', color=get_training_method_color('Baseline'), 
                   edgecolor='black', linewidth=0.5)
    bars2 = ax.bar(x - 1.0*width, sft_values, width, 
                   label='SFT', color=get_training_method_color('SFT'), 
                   edgecolor='black', linewidth=0.5)
    bars3 = ax.bar(x + 0.0*width, grpo_l_values, width, 
                   label='GRPO', color=get_training_method_color('GRPO'), 
                   edgecolor='black', linewidth=0.5)
    bars4 = ax.bar(x + 1.0*width, step_by_step_values, width, 
                   label='Step-by-step', color=get_training_method_color('Step-by-step'), 
                   edgecolor='black', linewidth=0.5)
    bars5 = ax.bar(x + 2.0*width, optimal_values, width,
                   label='Real Solution', color='#666666',  # Grey for reference
                   edgecolor='black', linewidth=0.5, hatch='//')
    
    # Customize axes
    ax.set_xlabel('Difficulty Level', fontsize=10)
    ax.set_ylabel('Average Path Length (steps)', fontsize=10)
    
    # Set x-axis ticks to difficulty levels
    ax.set_xticks(x)
    ax.set_xticklabels(['1', '2', '3', '4', '5'])
    
    # Add grid for readability (only horizontal)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5, axis='y')
    
    # Add legend below the plot with 2 columns
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.25), 
              ncol=2, frameon=False, framealpha=0.9, fontsize=9)
    
    # Set y-axis to start from 0 for better context
    ax.set_ylim(bottom=0)
    
    # Tight layout
    plt.tight_layout()
    
    # Save figure
    output_dir = Path(__file__).parent / "results" / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_base = output_dir / "step_length_by_difficulty"
    plt.savefig(f"{output_base}.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{output_base}.pdf", bbox_inches='tight')
    plt.savefig(f"{output_base}.svg", bbox_inches='tight')
    
    print(f"\n✓ Saved figures to {output_dir}/")
    print(f"  - step_length_by_difficulty.png")
    print(f"  - step_length_by_difficulty.pdf")
    print(f"  - step_length_by_difficulty.svg")
    print("="*80 + "\n")
    
    # Show plot
    plt.show()

if __name__ == "__main__":
    print("Generating step length by difficulty chart...")
    print("(Averaging across model sizes: 0.6B, 4B, 14B, 32B)\n")
    create_step_length_by_difficulty_chart()

