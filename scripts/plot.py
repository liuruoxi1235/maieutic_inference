#!/usr/bin/env python3
"""
Script to plot KL divergence results from multiple maieutic prompting experiments.

Looks for subdirectories containing 'kl_output.txt' files and plots the average
lines for each experiment on a single plot, now with error bars.

Usage: python plot_multi_kl_results.py
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from typing import List, Tuple, Dict
from pathlib import Path
from scipy import stats

def load_kl_data(kl_file_path: str) -> List[Tuple[List[float], int]]:
    """Load KL divergence data from file, skipping 'inf' values.
    Modified to replace values before semicolon with the first value after semicolon.
    Returns list of tuples: (kl_values, s_full_index) where s_full_index is the position of semicolon (-1 if none)."""
    kl_data = []
    
    with open(kl_file_path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            try:
                # Check for semicolon to identify s_full
                s_full_index = -1
                
                # Split by both comma and semicolon, keeping track of which separator was used
                parts = []
                current_part = ""
                for i, char in enumerate(line):
                    if char in ',;':
                        if current_part.strip():
                            parts.append(current_part.strip())
                        current_part = ""
                        if char == ';' and s_full_index == -1:
                            s_full_index = len(parts) - 1  # Index before the semicolon
                    else:
                        current_part += char
                if current_part.strip():
                    parts.append(current_part.strip())
                
                # Parse values
                kl_values = []
                for val in parts:
                    try:
                        float_val = float(val)
                        # Skip infinite values
                        if not np.isinf(float_val):
                            kl_values.append(float_val)
                    except ValueError:
                        # Skip values that can't be converted to float
                        continue
                
                # Modify the data: replace values before semicolon with first value after semicolon
                if kl_values and s_full_index >= 0 and s_full_index + 1 < len(kl_values):
                    # Get the first value after the semicolon
                    replacement_value = kl_values[s_full_index + 1]
                    
                    # Replace all values before and at the semicolon position
                    for i in range(s_full_index + 1):
                        kl_values[i] = replacement_value
                
                if kl_values:  # Only add if we have valid values
                    kl_data.append((kl_values, s_full_index))
                    
            except Exception as e:
                continue
    
    return kl_data

def compute_average_line_with_errors(kl_data: List[Tuple[List[float], int]], x_limit: int = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute the average line across all runs in the dataset with error bars.
    
    Args:
        kl_data: List of tuples containing KL values and semicolon indices
        x_limit: Maximum x value (number of scenarios) to consider. If None, use all data.
        
    Returns:
        Tuple of (x_values, y_means, y_errors) where y_errors are standard errors
    """
    if not kl_data:
        return np.array([]), np.array([]), np.array([])
    
    # Find maximum number of scenarios across all runs
    max_scenarios = max(len(run[0]) for run in kl_data)
    
    # Apply x_limit if specified
    if x_limit is not None:
        max_scenarios = min(max_scenarios, x_limit + 1)  # +1 because we want to include x_limit
    
    # Store data for averaging (pad with NaN for different lengths)
    all_scenario_data = []
    for run_data, _ in kl_data:
        # Truncate run_data if it exceeds x_limit
        if x_limit is not None and len(run_data) > x_limit + 1:
            run_data = run_data[:x_limit + 1]
        
        padded_run = run_data + [np.nan] * (max_scenarios - len(run_data))
        all_scenario_data.append(padded_run)
    
    # Calculate mean and standard error at each position
    all_scenario_data = np.array(all_scenario_data)
    
    x_values = []
    y_means = []
    y_errors = []
    
    for scenario_idx in range(max_scenarios):
        # Get all valid (non-NaN) values at this position
        values_at_position = all_scenario_data[:, scenario_idx]
        valid_values = values_at_position[~np.isnan(values_at_position)]
        
        if len(valid_values) > 0:
            mean_val = np.mean(valid_values)
            if len(valid_values) > 1:
                std_val = np.std(valid_values, ddof=1)  # Sample standard deviation
                std_err = std_val / np.sqrt(len(valid_values))
            else:
                std_err = 0
            
            x_values.append(scenario_idx)
            y_means.append(mean_val)
            y_errors.append(std_err)
    
    return np.array(x_values), np.array(y_means), np.array(y_errors)

def compute_average_line(kl_data: List[Tuple[List[float], int]], x_limit: int = None) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the average line across all runs in the dataset.
    
    Args:
        kl_data: List of tuples containing KL values and semicolon indices
        x_limit: Maximum x value (number of scenarios) to consider. If None, use all data.
    """
    x_values, y_means, _ = compute_average_line_with_errors(kl_data, x_limit)
    return x_values, y_means

def get_last_n_average(y_values: np.ndarray, n: int = 3) -> float:
    """Get the average of the last n values."""
    if len(y_values) == 0:
        return np.nan
    if len(y_values) <= n:
        return np.mean(y_values)
    return np.mean(y_values[-n:])

def create_individual_line_chart(kl_data: List[Tuple[List[float], int]], output_dir: str, exp_name: str):
    """Create line chart with individual runs and average for a single experiment."""
    
    zoom_configs = [
        ("", "", "kl_line_chart", None),
        ("_zoomed", " (Zoomed to 95th Percentile)", "kl_line_chart_zoomed", "percentile"),
        ("_zoomed_to_05", " (Zoomed to 0.5)", "kl_line_chart_zoomed_to_05", 0.5)
    ]

    for zoom_suffix, zoom_title, zoom_filename, zoom_type in zoom_configs:
        plt.figure(figsize=(12, 8))

        max_scenarios = max(len(run[0]) for run in kl_data) if kl_data else 0

        if zoom_type == "percentile":
            all_values = [val for run, _ in kl_data for val in run]
            y_max = np.percentile(all_values, 95) if all_values else 1.0
        elif zoom_type == 0.5:
            y_max = 0.5
        else:
            y_max = None

        # Plot individual runs
        for i, (run_data, s_full_idx) in enumerate(kl_data):
            x_values = list(range(len(run_data)))

            if s_full_idx >= 0 and s_full_idx < len(run_data) - 1:
                plt.plot(x_values[:s_full_idx+2], run_data[:s_full_idx+2], 
                         color='orange', alpha=0.3, linewidth=0.8, 
                         label='Before all B sampled' if i == 0 else "")
                plt.plot(x_values[s_full_idx+1:], run_data[s_full_idx+1:], 
                         'k-', alpha=0.3, linewidth=0.8, 
                         label='After all B sampled' if i == 0 else "")
            else:
                plt.plot(x_values, run_data, 'k-', alpha=0.3, linewidth=0.8, 
                         label='Individual runs' if i == 0 else "")

        # Compute and plot average line with error bars
        x_avg, y_avg, y_err = compute_average_line_with_errors(kl_data)
        if len(y_avg) > 0:
            plt.errorbar(x_avg, y_avg, yerr=y_err, 
                        color='red', linewidth=2, capsize=3, capthick=1, 
                        label=f'Average across runs (±SE)', zorder=10)

        plt.xlabel('Number of Scenarios', fontsize=12)
        plt.ylabel('KL Divergence', fontsize=12)
        plt.title(f'{exp_name} - KL Divergence vs Number of Scenarios{zoom_title}\n(Maieutic Prompting Results)', fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)

        if kl_data:
            all_values = [val for run, _ in kl_data for val in run]
            if y_max is not None:
                plt.ylim(0, y_max)
            else:
                plt.ylim(0, max(all_values) * 1.1)
            plt.xlim(-0.5, max_scenarios - 0.5)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{zoom_filename}.png'), dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(output_dir, f'{zoom_filename}.pdf'), bbox_inches='tight')
        plt.close()  # Close figure to free memory


def create_individual_scatter_plot(kl_data: List[Tuple[List[float], int]], output_dir: str, exp_name: str):
    """Create scatter plot with line of best fit and average KL line for a single experiment."""
    # Flatten all data points
    x_values = []
    y_values = []
    colors = []
    
    for run_data, s_full_idx in kl_data:
        for scenario_idx, kl_value in enumerate(run_data):
            x_values.append(scenario_idx)
            y_values.append(kl_value)
            # Color based on whether this is before or after s_full
            if s_full_idx >= 0 and scenario_idx <= s_full_idx:
                colors.append('orange')
            else:
                colors.append('blue')
    
    if not x_values:
        print(f"No data to plot scatter for {exp_name}")
        return
    
    x_values = np.array(x_values)
    y_values = np.array(y_values)
    
    # Create full, zoomed to 95th percentile, and zoomed to 0.5 versions
    zoom_configs = [
        ("", "", "kl_scatter_plot", None),
        ("_zoomed", " (Zoomed to 95th Percentile)", "kl_scatter_plot_zoomed", "percentile"),
        ("_zoomed_to_05", " (Zoomed to 0.5)", "kl_scatter_plot_zoomed_to_05", 0.5)
    ]
    
    for zoom_suffix, zoom_title, zoom_filename, zoom_type in zoom_configs:
        plt.figure(figsize=(12, 8))
        
        # Calculate y_max based on zoom type
        if zoom_type == "percentile":
            y_max = np.percentile(y_values, 95) if len(y_values) > 0 else 1.0
        elif zoom_type == 0.5:
            y_max = 0.5
        else:
            y_max = None
        
        # Create scatter plot with colors
        for color in ['orange', 'blue']:
            mask = [c == color for c in colors]
            if any(mask):
                label = 'Before all B sampled' if color == 'orange' else 'After all B sampled'
                plt.scatter(x_values[mask], y_values[mask], alpha=0.6, s=20, 
                           color=color, label=label)
        
        # Calculate and plot line of best fit
        if len(x_values) > 1:
            slope, intercept, r_value, p_value, std_err = stats.linregress(x_values, y_values)
            line_x = np.array([x_values.min(), x_values.max()])
            line_y = slope * line_x + intercept
            
            plt.plot(line_x, line_y, 'g-', linewidth=2, 
                    label=f'Line of best fit (R² = {r_value**2:.3f})')
            
            # Add equation as text
            equation = f'y = {slope:.4f}x + {intercept:.4f}'
            plt.text(0.05, 0.95, equation, transform=plt.gca().transAxes, 
                    fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Calculate and plot average KL line with error bars
        x_avg, y_avg, y_err = compute_average_line_with_errors(kl_data)
        if len(y_avg) > 0:
            plt.errorbar(x_avg, y_avg, yerr=y_err, 
                        color='red', linewidth=2, capsize=3, capthick=1,
                        label=f'Average KL with SE', zorder=10)
        
        plt.xlabel('Number of Scenarios', fontsize=12)
        plt.ylabel('KL Divergence', fontsize=12)
        plt.title(f'{exp_name} - KL Divergence Scatter Plot{zoom_title}\n(All data points from all runs)', fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Set axis limits
        if y_max is not None:
            plt.ylim(0, y_max)
        else:
            plt.ylim(0, max(y_values) * 1.1)
        plt.xlim(-0.5, max(x_values) + 0.5)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{zoom_filename}.png'), dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(output_dir, f'{zoom_filename}.pdf'), bbox_inches='tight')
        plt.close()  # Close figure to free memory


def print_individual_summary_statistics(kl_data: List[Tuple[List[float], int]], exp_name: str):
    """Print summary statistics for a single experiment."""
    if not kl_data:
        print(f"No data to analyze for {exp_name}")
        return
    
    # Flatten all values
    all_values = [val for run, _ in kl_data for val in run]
    
    print(f"\n" + "="*60)
    print(f"INDIVIDUAL SUMMARY STATISTICS - {exp_name}")
    print("="*60)
    print(f"Number of runs: {len(kl_data)}")
    print(f"Total data points: {len(all_values)}")
    
    # Count how many runs have s_full marker
    s_full_count = sum(1 for _, s_full_idx in kl_data if s_full_idx >= 0)
    print(f"Runs with all B values sampled: {s_full_count}/{len(kl_data)}")
    
    if all_values:
        print(f"Overall KL statistics:")
        print(f"  Mean: {np.mean(all_values):.6f}")
        print(f"  Median: {np.median(all_values):.6f}")
        print(f"  Std Dev: {np.std(all_values):.6f}")
        print(f"  Min: {np.min(all_values):.6f}")
        print(f"  Max: {np.max(all_values):.6f}")
        
        # Check for any remaining infinite values (shouldn't happen after filtering)
        inf_count = sum(1 for val in all_values if np.isinf(val))
        if inf_count > 0:
            print(f"  WARNING: {inf_count} infinite values found in data")
    
    # Per-run statistics
    run_lengths = [len(run) for run, _ in kl_data]
    print(f"\nScenarios per run:")
    print(f"  Min scenarios: {min(run_lengths)}")
    print(f"  Max scenarios: {max(run_lengths)}")
    print(f"  Avg scenarios: {np.mean(run_lengths):.1f}")
    
    # Initial vs final KL comparison
    initial_kls = [run[0] for run, _ in kl_data if len(run) > 0]
    final_kls = [run[-1] for run, _ in kl_data if len(run) > 0]
    
    if initial_kls and final_kls:
        print(f"\nInitial vs Final KL:")
        print(f"  Initial KL (mean): {np.mean(initial_kls):.6f}")
        print(f"  Final KL (mean): {np.mean(final_kls):.6f}")
        print(f"  Average improvement: {np.mean(initial_kls) - np.mean(final_kls):.6f}")
        if np.mean(initial_kls) > 0:
            print(f"  Improvement rate: {((np.mean(initial_kls) - np.mean(final_kls)) / np.mean(initial_kls) * 100):.1f}%")


def clean_experiment_name(exp_name: str) -> str:
    """Remove '_output' from experiment name if it exists."""
    if exp_name.endswith('_output'):
        return exp_name[:-7]  # Remove the last 7 characters ('_output')
    return exp_name


def create_individual_experiment_plots(experiments: Dict[str, List[Tuple[List[float], int]]], base_dir: str):
    """Create individual plots for each experiment in their respective subdirectories."""
    
    for exp_name, kl_data in experiments.items():
        print(f"\nCreating individual plots for {exp_name}...")
        
        # Find the original subdirectory name (with _output if it existed)
        base_path = Path(base_dir)
        original_name = None
        
        # Look for subdirectories that match either the cleaned name or cleaned name + "_output"
        for subdir in base_path.iterdir():
            if subdir.is_dir():
                if clean_experiment_name(subdir.name) == exp_name:
                    original_name = subdir.name
                    break
        
        if original_name is None:
            print(f"Warning: Could not find original subdirectory for {exp_name}")
            continue
        
        # Create plots subdirectory
        plots_dir = base_path / original_name / "plots"
        plots_dir.mkdir(exist_ok=True)
        
        print(f"  Output directory: {plots_dir}")
        
        # Print individual summary statistics
        print_individual_summary_statistics(kl_data, exp_name)
        
        # Create individual plots
        create_individual_line_chart(kl_data, str(plots_dir), exp_name)
        create_individual_scatter_plot(kl_data, str(plots_dir), exp_name)
        
        print(f"  ✅ Individual plots saved for {exp_name}")

def load_all_experiments(base_dir: str) -> Dict[str, List[Tuple[List[float], int]]]:
    """Load all experiments from subdirectories containing kl_output.txt files."""
    experiments = {}
    base_path = Path(base_dir)
    
    # Look for all subdirectories
    for subdir in sorted(base_path.iterdir()):
        if subdir.is_dir():
            kl_file = subdir / "kl_output.txt"
            if kl_file.exists():
                # Use cleaned name for display but keep original for file operations
                display_name = clean_experiment_name(subdir.name)
                print(f"Loading experiment: {display_name}")
                kl_data = load_kl_data(str(kl_file))
                if kl_data:
                    experiments[display_name] = kl_data
                    print(f"  Loaded {len(kl_data)} runs from {display_name}")
                else:
                    print(f"  Warning: No valid data in {display_name}")
    
    return experiments

def create_multi_experiment_plot(experiments: Dict[str, List[Tuple[List[float], int]]], 
                                output_dir: str, x_limit: int = None):
    """Create plots comparing average lines from multiple experiments with error bars.
    
    Args:
        experiments: Dictionary of experiment names to KL data
        output_dir: Directory to save output plots
        x_limit: Maximum x value (number of scenarios) to plot. If None, use all data.
    """
    
    if not experiments:
        print("No experiments to plot")
        return
    
    # Color palette for different experiments
    colors = plt.cm.tab10(np.linspace(0, 1, len(experiments)))
    
    # Create full, zoomed to 95th percentile, and zoomed to 0.5 versions
    zoom_configs = [
        ("", "", "multi_experiment_comparison", None),
        ("_zoomed", " (Zoomed to 95th Percentile)", "multi_experiment_comparison_zoomed", "percentile"),
        ("_zoomed_to_05", " (Zoomed to 0.5)", "multi_experiment_comparison_zoomed_to_05", 0.5)
    ]
    
    # Collect all y values for percentile calculation
    all_y_values = []
    experiment_results = {}
    
    # Pre-compute all average lines with errors and x_limit applied
    for exp_name, kl_data in experiments.items():
        x_avg, y_avg, y_err = compute_average_line_with_errors(kl_data, x_limit)
        if len(y_avg) > 0:
            experiment_results[exp_name] = (x_avg, y_avg, y_err)
            all_y_values.extend(y_avg)
    
    # Add x_limit info to title if specified
    x_limit_text = f" (x ≤ {x_limit})" if x_limit is not None else ""
    
    for zoom_suffix, zoom_title, zoom_filename, zoom_type in zoom_configs:
        plt.figure(figsize=(14, 8))
        
        # Calculate y_max based on zoom type
        if zoom_type == "percentile":
            y_max = np.percentile(all_y_values, 95) if all_y_values else 1.0
        elif zoom_type == 0.5:
            y_max = 0.5
        else:
            y_max = None
        
        # Plot each experiment's average line with error bars
        for (exp_name, (x_avg, y_avg, y_err)), color in zip(experiment_results.items(), colors):
            # Calculate average of last 3 points
            last_3_avg = get_last_n_average(y_avg, 3)
            
            # Create label with last 3 average (using cleaned name)
            label = f'{exp_name} (last 3 avg: {last_3_avg:.4f})'
            
            # Plot the line with error bars
            plt.errorbar(x_avg, y_avg, yerr=y_err, 
                        linewidth=2, color=color, label=label,
                        capsize=3, capthick=1, alpha=0.8)
            
            # Add text annotation at the end of the line
            if len(x_avg) > 0:
                # Position text slightly to the right of the last point
                text_x = x_avg[-1] + 0.5
                text_y = y_avg[-1]
                
                # Adjust text position if it would be cut off by zoom
                if y_max is not None and text_y > y_max * 0.95:
                    text_y = y_max * 0.95
                
                plt.text(text_x, text_y, f'{last_3_avg:.4f}', 
                        fontsize=9, color=color, 
                        verticalalignment='center',
                        bbox=dict(boxstyle='round,pad=0.3', 
                                facecolor='white', 
                                edgecolor=color,
                                alpha=0.7))
        
        plt.xlabel('Number of Scenarios', fontsize=12)
        plt.ylabel('KL Divergence', fontsize=12)
        plt.title(f'Multi-Experiment KL Divergence Comparison{zoom_title}{x_limit_text}\n(Average lines with standard error)', 
                 fontsize=14)
        plt.legend(loc='best', fontsize=10)
        plt.grid(True, alpha=0.3)
        
        # Set axis limits
        if experiment_results:
            if x_limit is not None:
                plt.xlim(-0.5, x_limit + 2)  # Extra space for text annotations
            else:
                max_x = max(x_avg[-1] for x_avg, _, _ in experiment_results.values())
                plt.xlim(-0.5, max_x + 2)  # Extra space for text annotations
            
            if y_max is not None:
                plt.ylim(0, y_max)
            else:
                plt.ylim(0, max(all_y_values) * 1.1 if all_y_values else 1.0)
        
        plt.tight_layout()
        
        # Save the plots
        png_path = os.path.join(output_dir, f'{zoom_filename}.png')
        pdf_path = os.path.join(output_dir, f'{zoom_filename}.pdf')
        plt.savefig(png_path, dpi=300, bbox_inches='tight')
        plt.savefig(pdf_path, bbox_inches='tight')
        print(f"Saved: {png_path}")
        print(f"Saved: {pdf_path}")
        plt.show()

def create_pairwise_comparison_plots(experiments: Dict[str, List[Tuple[List[float], int]]], base_dir: str):
    """Create pairwise comparison scatter plots between all experiment pairs."""
    
    exp_names = list(experiments.keys())
    n_experiments = len(exp_names)
    
    if n_experiments < 2:
        print("Need at least 2 experiments for pairwise comparison")
        return
    
    # Create pairwise comparison directory
    pairwise_dir = os.path.join(base_dir, "pairwise_comparisons")
    if not os.path.exists(pairwise_dir):
        print(f"Creating pairwise comparison directory: {pairwise_dir}")
        os.makedirs(pairwise_dir)
    
    print(f"\nCreating pairwise comparison plots for {n_experiments} experiments...")
    print(f"Total pairs to create: {n_experiments * (n_experiments - 1) // 2}")
    print(f"Pairwise plots directory: {pairwise_dir}")
    
    # Generate colors for different lines
    max_lines = max(len(kl_data) for kl_data in experiments.values())
    colors = plt.cm.tab20(np.linspace(0, 1, min(max_lines, 20)))  # Use up to 20 colors
    
    pair_count = 0
    for i in range(n_experiments):
        for j in range(i + 1, n_experiments):
            exp1_name = exp_names[i]
            exp2_name = exp_names[j]
            exp1_data = experiments[exp1_name]
            exp2_data = experiments[exp2_name]
            
            pair_count += 1
            print(f"  Creating pair {pair_count}: {exp1_name} vs {exp2_name}")
            
            # Create subdirectory for this pair
            safe_exp1_name = exp1_name.replace('/', '_').replace(' ', '_')
            safe_exp2_name = exp2_name.replace('/', '_').replace(' ', '_')
            pair_subdir = os.path.join(pairwise_dir, f"{safe_exp1_name}_vs_{safe_exp2_name}")
            os.makedirs(pair_subdir, exist_ok=True)
            
            # Create the pairwise comparison plots (all zoom levels)
            create_single_pairwise_plot(exp1_data, exp2_data, exp1_name, exp2_name, 
                                      pair_subdir, colors)

def create_single_pairwise_plot(exp1_data: List[Tuple[List[float], int]], 
                               exp2_data: List[Tuple[List[float], int]],
                               exp1_name: str, exp2_name: str, 
                               output_dir: str, colors):
    """Create a single pairwise comparison scatter plot between two experiments with multiple zoom levels."""
    
    # Determine the number of lines to consider (minimum of both experiments)
    max_lines = min(len(exp1_data), len(exp2_data))
    
    if max_lines == 0:
        print(f"    Warning: No data to compare between {exp1_name} and {exp2_name}")
        return
    
    # Collect all data points for zoom calculations
    all_x_values = []
    all_y_values = []
    line_data = []  # Store data for each line for plotting
    
    # Process each line pair to collect data
    for line_idx in range(max_lines):
        exp1_line, exp1_semicolon = exp1_data[line_idx]
        exp2_line, exp2_semicolon = exp2_data[line_idx]
        
        # Determine the start index (after semicolon for both)
        exp1_start = exp1_semicolon + 1 if exp1_semicolon >= 0 else 0
        exp2_start = exp2_semicolon + 1 if exp2_semicolon >= 0 else 0
        
        # Use the later start index to ensure both are after their respective semicolons
        start_idx = max(exp1_start, exp2_start)
        
        # Determine end index (minimum length available from start)
        exp1_available = len(exp1_line) - start_idx
        exp2_available = len(exp2_line) - start_idx
        end_length = min(exp1_available, exp2_available)
        
        if end_length <= 0:
            line_data.append(([], []))  # Empty data for this line
            continue
        
        # Extract the comparable portions
        exp1_values = exp1_line[start_idx:start_idx + end_length]
        exp2_values = exp2_line[start_idx:start_idx + end_length]
        
        line_data.append((exp1_values, exp2_values))
        all_x_values.extend(exp1_values)
        all_y_values.extend(exp2_values)
    
    if not all_x_values or not all_y_values:
        print(f"    Warning: No valid data points for {exp1_name} vs {exp2_name}")
        return
    
    # Calculate zoom ranges
    x_95th = np.percentile(all_x_values, 95)
    y_95th = np.percentile(all_y_values, 95)
    zoom_95_max = max(x_95th, y_95th)
    zoom_05_max = 0.5
    
    # Define zoom configurations
    zoom_configs = [
        ("full", "Full Range", None),
        ("95th", "95th Percentile Zoom", zoom_95_max),
        ("05", "0.5 Zoom", zoom_05_max)
    ]
    
    total_points = len(all_x_values)
    
    # Create plots for each zoom level
    for zoom_suffix, zoom_title, zoom_max in zoom_configs:
        plt.figure(figsize=(12, 10))
        
        points_in_view = 0
        
        # Plot each line's data
        for line_idx, (exp1_values, exp2_values) in enumerate(line_data):
            if len(exp1_values) > 0 and len(exp2_values) > 0:
                # Choose color (cycle through available colors)
                color = colors[line_idx % len(colors)]
                
                # Filter points based on zoom level
                if zoom_max is not None:
                    # Only plot points within zoom range
                    filtered_x = []
                    filtered_y = []
                    for x, y in zip(exp1_values, exp2_values):
                        if x <= zoom_max and y <= zoom_max:
                            filtered_x.append(x)
                            filtered_y.append(y)
                    plot_x, plot_y = filtered_x, filtered_y
                else:
                    plot_x, plot_y = exp1_values, exp2_values
                
                if len(plot_x) > 0:
                    plt.scatter(plot_x, plot_y, 
                               color=color, alpha=0.7, s=30, 
                               label=f'Line {line_idx + 1} ({len(plot_x)} points)')
                    points_in_view += len(plot_x)
        
        # Add diagonal line (y = x) for reference
        if points_in_view > 0:
            if zoom_max is not None:
                plt.plot([0, zoom_max], [0, zoom_max], 
                        'k--', alpha=0.5, linewidth=1, label='y = x (equal performance)')
                plt.xlim(0, zoom_max)
                plt.ylim(0, zoom_max)
            else:
                max_val = max(max(all_x_values), max(all_y_values))
                plt.plot([0, max_val], [0, max_val], 
                        'k--', alpha=0.5, linewidth=1, label='y = x (equal performance)')
                plt.xlim(0, max_val * 1.05)
                plt.ylim(0, max_val * 1.05)
        
        plt.xlabel(f'{exp1_name} KL Divergence', fontsize=12)
        plt.ylabel(f'{exp2_name} KL Divergence', fontsize=12)
        plt.title(f'Pairwise Comparison: {exp1_name} vs {exp2_name} ({zoom_title})\n'
                  f'(Points after semicolon, {max_lines} lines compared, {points_in_view}/{total_points} points shown)', 
                  fontsize=14)
        
        # Add legend
        if points_in_view > 0:
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
        
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save the plot
        filename_base = f'pairwise_comparison_{zoom_suffix}'
        png_path = os.path.join(output_dir, f'{filename_base}.png')
        pdf_path = os.path.join(output_dir, f'{filename_base}.pdf')
        
        plt.savefig(png_path, dpi=300, bbox_inches='tight')
        plt.savefig(pdf_path, bbox_inches='tight')
        plt.close()
        
        print(f"    Saved {zoom_title}: {png_path}")
    
    print(f"    Total points: {total_points} from {max_lines} lines")

def print_experiment_summary(experiments: Dict[str, List[Tuple[List[float], int]]], x_limit: int = None):
    """Print summary statistics for all experiments.
    
    Args:
        experiments: Dictionary of experiment names to KL data
        x_limit: Maximum x value (number of scenarios) to consider in statistics
    """
    print("\n" + "="*60)
    print("EXPERIMENT SUMMARY")
    if x_limit is not None:
        print(f"(Statistics computed for x ≤ {x_limit})")
    print("="*60)
    print(f"Total experiments found: {len(experiments)}")
    
    for exp_name, kl_data in experiments.items():
        print(f"\n{exp_name}:")
        print(f"  Number of runs: {len(kl_data)}")
        
        # Get average line with x_limit applied
        x_avg, y_avg = compute_average_line(kl_data, x_limit)
        if len(y_avg) > 0:
            print(f"  Average line statistics:")
            print(f"    Length: {len(y_avg)} scenarios")
            print(f"    Initial KL: {y_avg[0]:.6f}")
            print(f"    Final KL: {y_avg[-1]:.6f}")
            print(f"    Last 3 average: {get_last_n_average(y_avg, 3):.6f}")
            print(f"    Min KL: {np.min(y_avg):.6f}")
            print(f"    Max KL: {np.max(y_avg):.6f}")
            print(f"    Improvement: {y_avg[0] - y_avg[-1]:.6f}")
            if y_avg[0] > 0:
                print(f"    Improvement rate: {((y_avg[0] - y_avg[-1]) / y_avg[0] * 100):.1f}%")

def main():
    # Define paths directly in the script
    base_dir = "/home/rliu79/reasoning/experiment_maieutic_compare"
    output_dir = "/home/rliu79/reasoning/experiment_maieutic_compare/combined_plots"

    # Define x-axis limit (set to None for no limit)
    x_limit = 50  # Only consider data points where x <= 50. Set to None to use all data.
    
    # Validate base directory
    if not os.path.exists(base_dir):
        print(f"Error: Base directory not found: {base_dir}")
        return
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        print(f"Creating output directory: {output_dir}")
        os.makedirs(output_dir)
    
    print(f"Base directory: {base_dir}")
    print(f"Output directory: {output_dir}")
    if x_limit is not None:
        print(f"X-axis limit: {x_limit} (only considering scenarios 0 to {x_limit})")
    else:
        print(f"X-axis limit: None (using all data)")
    print(f"\nSearching for experiments with kl_output.txt files...")
    
    # Load all experiments
    experiments = load_all_experiments(base_dir)
    
    if not experiments:
        print("No experiments found with kl_output.txt files")
        return
    
    # Print summary with x_limit applied
    print_experiment_summary(experiments, x_limit)
    
    # Create individual plots for each experiment
    print(f"\nCreating individual plots for each experiment...")
    create_individual_experiment_plots(experiments, base_dir)
    
    # Create plots with x_limit applied
    print(f"\nCreating multi-experiment comparison plots...")
    create_multi_experiment_plot(experiments, output_dir, x_limit)

    print(f"\nCreating pairwise comparison plots...")
    create_pairwise_comparison_plots(experiments, base_dir)
    
    print(f"\n✅ All plots saved to: {output_dir}")

if __name__ == "__main__":
    main()