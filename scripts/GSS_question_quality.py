import json
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import os
import pandas as pd
from scipy.stats import entropy

class InformationGainKLPlotter:
    def __init__(self, json_file, csv_file, output_dir):
        self.json_file = json_file
        self.csv_file = csv_file
        self.output_dir = output_dir
        self.df = None
        
        # Define responses to exclude (from your CSV processing code)
        self.INVALID_RESPONSES = {"don't know", "iap", "not available in this year", "no answer", "skipped on web", "refused"}
        
    def load_data(self):
        """Load both JSON and CSV files"""
        print(f"Loading JSON data from: {self.json_file}")
        with open(self.json_file, 'r') as f:
            self.json_data = json.load(f)
        
        print(f"Loading CSV data from: {self.csv_file}")
        self.df = pd.read_csv(self.csv_file, low_memory=False)
        
        print(f"✓ Loaded {len(self.json_data)} variables from JSON file")
        print(f"✓ Loaded CSV with {len(self.df)} rows and {len(self.df.columns)} columns")
    
    def bin_numeric_series(self, series, n_bins=5):
        """Bin numeric series into n_bins, handling cases with limited unique values"""
        min_val = int(series.min())
        max_val = int(series.max())
        
        # Check if we have very few unique values
        unique_values = sorted(series.unique())
        n_unique = len(unique_values)
        
        # If we have fewer unique values than desired bins, use unique values as boundaries
        if n_unique <= n_bins:
            # Create boundaries based on actual unique values
            boundaries = []
            for i, val in enumerate(unique_values):
                boundaries.append(val)
            boundaries.append(max_val + 1)  # Add upper boundary
            
            # Create labels for each unique value
            labels = []
            for i in range(len(boundaries) - 1):
                if i == len(boundaries) - 2:  # Last bin
                    labels.append(f"{boundaries[i]}")
                else:
                    labels.append(f"{boundaries[i]}")
        else:
            # Original binning logic for cases with many unique values
            boundaries = np.linspace(min_val, max_val + 1, n_bins + 1)
            boundaries = np.round(boundaries).astype(int)
            
            # Remove duplicates while preserving order
            unique_boundaries = []
            for b in boundaries:
                if not unique_boundaries or b != unique_boundaries[-1]:
                    unique_boundaries.append(b)
            boundaries = unique_boundaries
            
            # If we still don't have enough boundaries, fall back to unique values
            if len(boundaries) - 1 < 2:
                boundaries = list(range(min_val, max_val + 2))
            
            labels = []
            for i in range(len(boundaries) - 1):
                lower = boundaries[i]
                if i < len(boundaries) - 2:
                    upper = boundaries[i+1] - 1
                else:
                    upper = boundaries[i+1] - 1
                    
                if lower == upper:
                    labels.append(f"{lower}")
                else:
                    labels.append(f"{lower} - {upper}")
        
        try:
            binned_series = pd.cut(series, bins=boundaries, labels=labels, include_lowest=True, right=False, duplicates='drop')
            return binned_series
        except Exception as e:
            # If binning still fails, treat as categorical
            print(f"    Binning failed for numeric data, treating as categorical: {e}")
            return series.astype(str)
    
    def get_variable_distribution_with_constraints(self, variable, constraints):
        """Get distribution of variable given constraints (adapted from your CSV code)"""
        # Validate constraints
        for c_var, c_val in constraints.items():
            if c_var not in self.df.columns:
                raise ValueError(f"Constraint variable '{c_var}' not found in the DataFrame.")
            domain = set(self.df[c_var].dropna().unique())
            if isinstance(c_val, list):
                for val in c_val:
                    if val not in domain:
                        raise ValueError(f"Value '{val}' for constraint variable '{c_var}' is not in its domain")
            else:
                if c_val not in domain:
                    raise ValueError(f"Value '{c_val}' for constraint variable '{c_var}' is not in its domain")
        
        # Filter DataFrame based on constraints
        filtered_df = self.df.copy()
        for c_var, c_val in constraints.items():
            if isinstance(c_val, list):
                filtered_df = filtered_df[filtered_df[c_var].isin(c_val)]
            else:
                filtered_df = filtered_df[filtered_df[c_var] == c_val]
        
        if variable not in filtered_df.columns:
            raise ValueError(f"Variable '{variable}' not found in the DataFrame.")
        
        valid_series = filtered_df[variable][~filtered_df[variable].isin(self.INVALID_RESPONSES)]
        
        # Check if we have enough data points
        if len(valid_series) < 10:  # Minimum threshold for meaningful statistics
            raise ValueError(f"Insufficient data: only {len(valid_series)} valid responses after applying constraints")
        
        # Convert to numeric and check if mostly numeric
        numeric_initial = pd.to_numeric(valid_series, errors='coerce').reindex(valid_series.index)
        mask = numeric_initial.notnull() & numeric_initial.between(0, 5000)
        numeric_values = numeric_initial[mask]
        
        n_total = len(valid_series)
        n_numeric = mask.sum()
        frac_numeric = n_numeric / n_total if n_total > 0 else 0
        
        if frac_numeric >= 0.5:
            numeric_series = numeric_values.dropna()
            if len(numeric_series) < 5:
                raise ValueError(f"Insufficient numeric data: only {len(numeric_series)} numeric values")
            
            # Check if we have enough unique values for meaningful binning
            unique_values = len(numeric_series.unique())
            if unique_values < 2:
                raise ValueError(f"Insufficient diversity: only {unique_values} unique values")
            
            binned = self.bin_numeric_series(numeric_series, n_bins=min(5, unique_values))
            distribution = binned.value_counts(normalize=True, dropna=False)
        else:
            distribution = valid_series.value_counts(normalize=True, dropna=False)
            
            # Check diversity for categorical data too
            if len(distribution) < 2:
                raise ValueError(f"Insufficient diversity: only {len(distribution)} unique categories")
        
        return distribution
    
    def calculate_entropy(self, distribution):
        """Calculate entropy of a probability distribution"""
        # Remove any zero probabilities to avoid log(0)
        probs = distribution.values
        probs = probs[probs > 0]
        return entropy(probs, base=2)
    
    def calculate_conditional_probability(self, additional_constraints, original_constraints):
        """Calculate P(additional conditions | original conditions)"""
        try:
            # Get count with original constraints only
            original_filtered = self.df.copy()
            for c_var, c_val in original_constraints.items():
                if isinstance(c_val, list):
                    original_filtered = original_filtered[original_filtered[c_var].isin(c_val)]
                else:
                    original_filtered = original_filtered[original_filtered[c_var] == c_val]
            
            # Get count with both original and additional constraints
            combined_constraints = {**original_constraints, **additional_constraints}
            combined_filtered = self.df.copy()
            for c_var, c_val in combined_constraints.items():
                if isinstance(c_val, list):
                    combined_filtered = combined_filtered[combined_filtered[c_var].isin(c_val)]
                else:
                    combined_filtered = combined_filtered[combined_filtered[c_var] == c_val]
            
            # Calculate conditional probability
            count_original = len(original_filtered)
            count_combined = len(combined_filtered)
            
            if count_original > 0:
                return count_combined / count_original
            else:
                return 0.0
                
        except (ValueError, KeyError) as e:
            print(f"  Warning: Could not calculate conditional probability: {e}")
            return 0.0
    
    def calculate_information_metrics(self, var_name, original_constraints, additional_constraints):
        """Calculate both information gain and weighted information gain"""
        try:
            # Get H(target | original conditions)
            dist_original = self.get_variable_distribution_with_constraints(var_name, original_constraints)
            h_original = self.calculate_entropy(dist_original)
            
            # Combine constraints for H(target | original + additional)
            combined_constraints = {**original_constraints, **additional_constraints}
            dist_combined = self.get_variable_distribution_with_constraints(var_name, combined_constraints)
            h_combined = self.calculate_entropy(dist_combined)
            
            # Information gain is the reduction in entropy
            info_gain = h_original - h_combined
            
            # Calculate conditional probability P(additional | original)
            cond_prob = self.calculate_conditional_probability(additional_constraints, original_constraints)
            
            # Weighted information gain
            weighted_info_gain = info_gain * cond_prob
            
            return info_gain, weighted_info_gain
            
        except (ValueError, KeyError) as e:
            print(f"  Warning: Could not calculate information metrics for {var_name}: {e}")
            return None, None
    
    def parse_constraints_from_prompt(self, prompt_data):
        """
        Parse constraints from prompt data using the conditions_used field.
        Returns a list of constraint dictionaries in the order they appear.
        """
        constraints = []
        
        # Use conditions_used field
        if 'conditions_used' in prompt_data and prompt_data['conditions_used']:
            conditions_used = prompt_data['conditions_used']
            # Convert to list of single-key dictionaries to maintain order
            for var, value in conditions_used.items():
                constraints.append({var: value})
        
        return constraints
    
    def extract_plotting_data(self):
        """Extract data points for plotting from JSON file"""
        all_points = []
        
        for var_name, var_data in self.json_data.items():
            # Check for both types of prompts
            has_rand = 'rand_prompts' in var_data and var_data['rand_prompts'] is not None
            has_relevant = 'relevant_prompts' in var_data and var_data['relevant_prompts'] is not None
            
            # Skip if both or neither are present
            if (has_rand and has_relevant) or (not has_rand and not has_relevant):
                continue
            
            prompts_field = 'rand_prompts' if has_rand else 'relevant_prompts'
            prompts_data = var_data[prompts_field]
            
            for prompt_data in prompts_data:
                if 'KL' not in prompt_data or 'num_conditions' not in prompt_data:
                    continue
                
                kl_value = prompt_data['KL']
                num_conditions = prompt_data['num_conditions']
                
                # Skip if no conditions (can't split into original vs additional)
                if num_conditions == 0:
                    continue
                
                # Extract constraints from the conditions_used field
                try:
                    all_constraints = self.parse_constraints_from_prompt(prompt_data)
                    
                    # Verify we have the expected number of constraints
                    if len(all_constraints) != num_conditions:
                        print(f"  Warning: Constraint count mismatch for {var_name}. "
                              f"Expected {num_conditions}, found {len(all_constraints)}")
                        continue
                    
                    # Split constraints: last one is "additional", rest are "original"
                    if len(all_constraints) >= 1:
                        # Original conditions: all except the last
                        original_constraints = {}
                        for constraint_dict in all_constraints[:-1]:
                            original_constraints.update(constraint_dict)
                        
                        # Additional conditions: just the last one
                        additional_constraints = all_constraints[-1]
                        
                        # The number of original conditions
                        num_original_conditions = len(all_constraints) - 1
                        
                        # Calculate both information metrics
                        info_gain, weighted_info_gain = self.calculate_information_metrics(
                            var_name, original_constraints, additional_constraints
                        )
                        
                        # Calculate H(target | original conditions) for coloring
                        try:
                            dist_original = self.get_variable_distribution_with_constraints(var_name, original_constraints)
                            h_original = self.calculate_entropy(dist_original)
                        except (ValueError, KeyError) as e:
                            print(f"  Warning: Could not calculate original entropy for {var_name}: {e}")
                            h_original = None
                        
                        if info_gain is not None and weighted_info_gain is not None and h_original is not None:
                            all_points.append({
                                'var_name': var_name,
                                'num_original_conditions': num_original_conditions,
                                'info_gain': info_gain,
                                'weighted_info_gain': weighted_info_gain,
                                'kl_value': kl_value,
                                'h_original': h_original,  # Use this for coloring instead of avg_kl
                                'prompt_type': prompts_field,
                                'original_constraints': original_constraints,
                                'additional_constraints': additional_constraints
                            })
                            
                            # Debug output
                            print(f"  {var_name} ({num_original_conditions} orig): "
                                  f"Info gain = {info_gain:.4f}, "
                                  f"Weighted = {weighted_info_gain:.4f}, "
                                  f"KL = {kl_value:.4f}, "
                                  f"H(orig) = {h_original:.4f}")
                        else:
                            print(f"  Warning: Could not calculate information metrics or original entropy for {var_name}")
                    else:
                        print(f"  Warning: No constraints found for {var_name}")
                        
                except Exception as e:
                    print(f"  Warning: Could not process prompt for {var_name}: {e}")
                    continue
        
        print(f"Extracted {len(all_points)} data points for plotting")
        return all_points
    
    def create_condition_specific_plots(self, all_points, use_weighted=False):
        """Create plots for each number of original conditions (0-4)"""
        plot_type = "weighted" if use_weighted else "unweighted"
        output_subdir = os.path.join(self.output_dir, plot_type)
        os.makedirs(output_subdir, exist_ok=True)
        
        condition_stats = {}
        
        # Determine which metric to use
        x_key = 'weighted_info_gain' if use_weighted else 'info_gain'
        x_label = 'Weighted Information Gain (H(target|original) - H(target|original+more)) * P(more|original)' if use_weighted else 'Information Gain (H(target|original) - H(target|original+more))'
        
        # Create plot for each number of original conditions
        for num_conditions in range(5):  # 0, 1, 2, 3, 4
            # Filter points for this number of conditions
            condition_points = [p for p in all_points if p['num_original_conditions'] == num_conditions]
            
            if len(condition_points) < 2:
                print(f"Skipping {num_conditions} conditions ({plot_type}): insufficient data points ({len(condition_points)})")
                continue
            
            # Extract data for plotting
            x_values = [p[x_key] for p in condition_points]
            kl_values = [p['kl_value'] for p in condition_points]
            colors = [p['h_original'] for p in condition_points]  # Use original entropy for coloring
            
            # Filter out extreme KL values
            filtered_indices = [i for i, kl in enumerate(kl_values) if kl <= 2]
            filtered_x_values = [x_values[i] for i in filtered_indices]
            filtered_kl_values = [kl_values[i] for i in filtered_indices]
            filtered_colors = [colors[i] for i in filtered_indices]
            filtered_count = len(kl_values) - len(filtered_kl_values)
            
            if len(filtered_x_values) < 2:
                print(f"Skipping {num_conditions} conditions ({plot_type}): insufficient data after filtering ({len(filtered_x_values)})")
                continue
            
            # Create the plot
            plt.figure(figsize=(12, 8))
            
            # Create scatter plot with color mapping
            scatter = plt.scatter(filtered_x_values, filtered_kl_values, c=filtered_colors, 
                                alpha=0.7, s=80, cmap='viridis')
            
            # Add colorbar
            cbar = plt.colorbar(scatter)
            cbar.set_label('H(target | original conditions)', fontsize=12)
            
            # Calculate correlation and line of best fit
            if len(filtered_x_values) > 1:
                correlation = np.corrcoef(filtered_x_values, filtered_kl_values)[0, 1]
                slope, intercept, r_value, p_value, std_err = stats.linregress(filtered_x_values, filtered_kl_values)
                
                # Plot line of best fit
                x_line = np.linspace(min(filtered_x_values), max(filtered_x_values), 100)
                y_line = slope * x_line + intercept
                plt.plot(x_line, y_line, 'r-', alpha=0.8, linewidth=2,
                        label=f'Best fit (R² = {r_value**2:.3f})')
                
                condition_stats[num_conditions] = {
                    'correlation': correlation,
                    'r_squared': r_value**2,
                    'slope': slope,
                    'intercept': intercept,
                    'n_points': len(filtered_x_values),
                    'filtered_points': filtered_count
                }
            
            plt.xlabel(x_label, fontsize=12)
            plt.ylabel('KL Divergence', fontsize=12)
            
            title_prefix = "Weighted Information Gain" if use_weighted else "Information Gain"
            plt.title(f'{title_prefix} vs KL Divergence: {num_conditions} Original Condition{"s" if num_conditions != 1 else ""}\n(Colored by H(target|original), KL ≤ 2)', fontsize=14)
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Add statistics text
            stats_text = f'Points: {len(filtered_x_values)}'
            if filtered_count > 0:
                stats_text += f' (filtered: {filtered_count})'
            if len(filtered_x_values) > 1:
                stats_text += f'\nCorrelation: {correlation:.3f}'
            if filtered_colors:
                stats_text += f'\nAvg H(original): {np.mean(filtered_colors):.3f}'
                
            plt.text(0.02, 0.98, stats_text,
                    transform=plt.gca().transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            # Save the plot
            output_path = os.path.join(output_subdir, f'{plot_type}_info_gain_vs_kl_{num_conditions}_conditions.png')
            plt.tight_layout()
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Created {plot_type} plot for {num_conditions} condition{'s' if num_conditions != 1 else ''}: {len(filtered_x_values)} points")
        
        return condition_stats
    
    def create_combined_plot(self, all_points, use_weighted=False):
        """Create a combined plot with all condition sizes"""
        if not all_points:
            print(f"No data points for combined {'weighted' if use_weighted else 'unweighted'} plot")
            return
        
        plot_type = "weighted" if use_weighted else "unweighted"
        output_subdir = os.path.join(self.output_dir, plot_type)
        
        # Filter out points with KL > 2
        filtered_points = [p for p in all_points if p['kl_value'] <= 2]
        filtered_out_count = len(all_points) - len(filtered_points)
        
        if filtered_out_count > 0:
            print(f"  Filtered out {filtered_out_count} points with KL > 2 for combined {plot_type} plot")
        
        if not filtered_points:
            print(f"No data points remaining after filtering for combined {plot_type} plot")
            return
        
        # Determine which metric to use
        x_key = 'weighted_info_gain' if use_weighted else 'info_gain'
        x_label = 'Weighted Information Gain (H(target|original) - H(target|original+more)) * P(more|original)' if use_weighted else 'Information Gain (H(target|original) - H(target|original+more))'
        
        # Extract data for plotting
        x_values = [p[x_key] for p in filtered_points]
        kl_values = [p['kl_value'] for p in filtered_points]
        colors = [p['h_original'] for p in filtered_points]  # Use original entropy for coloring
        
        plt.figure(figsize=(12, 10))
        
        # Create scatter plot with color mapping
        scatter = plt.scatter(x_values, kl_values, c=colors, alpha=0.6, s=60, cmap='viridis')
        
        # Add colorbar
        cbar = plt.colorbar(scatter)
        cbar.set_label('H(target | original conditions)', fontsize=12)
        
        # Calculate overall correlation and line of best fit
        if len(x_values) > 1:
            correlation = np.corrcoef(x_values, kl_values)[0, 1]
            slope, intercept, r_value, p_value, std_err = stats.linregress(x_values, kl_values)
            
            # Plot line of best fit
            x_line = np.linspace(min(x_values), max(x_values), 100)
            y_line = slope * x_line + intercept
            plt.plot(x_line, y_line, 'r-', alpha=0.8, linewidth=2,
                    label=f'Best fit (R² = {r_value**2:.3f})')
        
        plt.xlabel(x_label, fontsize=12)
        plt.ylabel('KL Divergence', fontsize=12)
        
        title_prefix = "Weighted Information Gain" if use_weighted else "Information Gain"
        plt.title(f'{title_prefix} vs KL Divergence: All Original Condition Sizes Combined\n(Colored by H(target|original), KL ≤ 2)', fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Add statistics text
        stats_text = f'Points: {len(x_values)}'
        if filtered_out_count > 0:
            stats_text += f' (filtered: {filtered_out_count})'
        if len(x_values) > 1:
            stats_text += f'\nCorrelation: {correlation:.3f}'
        if colors:
            stats_text += f'\nAvg H(original): {np.mean(colors):.3f}'
        
        # Count points by condition size
        condition_counts = {}
        for point in filtered_points:
            condition = point['num_original_conditions']
            condition_counts[condition] = condition_counts.get(condition, 0) + 1
        
        condition_breakdown = ', '.join([f'{k}: {v}' for k, v in sorted(condition_counts.items())])
        stats_text += f'\nBy condition: {condition_breakdown}'
            
        plt.text(0.02, 0.98, stats_text,
                transform=plt.gca().transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Save the plot
        output_path = os.path.join(output_subdir, f'{plot_type}_info_gain_vs_kl_combined.png')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Created combined {plot_type} plot with {len(x_values)} total points")
        
        return {
            'correlation': correlation if len(x_values) > 1 else None,
            'r_squared': r_value**2 if len(x_values) > 1 else None,
            'slope': slope if len(x_values) > 1 else None,
            'intercept': intercept if len(x_values) > 1 else None,
            'n_points': len(x_values),
            'filtered_points': filtered_out_count
        }
    
    def run_analysis(self):
        """Run the full information gain analysis"""
        print("Information Gain vs KL Divergence Analysis")
        print("=" * 50)
        
        # Load data
        self.load_data()
        
        # Extract plotting data
        print(f"\nExtracting plotting data...")
        all_points = self.extract_plotting_data()
        
        if not all_points:
            print("❌ No valid data points found for plotting")
            return
        
        # Create both sets of plots
        print(f"\n📊 Creating UNWEIGHTED plots (H(target|original) - H(target|original+more))...")
        unweighted_condition_stats = self.create_condition_specific_plots(all_points, use_weighted=False)
        unweighted_combined_stats = self.create_combined_plot(all_points, use_weighted=False)
        
        print(f"\n📊 Creating WEIGHTED plots ((H(target|original) - H(target|original+more)) * P(more|original))...")
        weighted_condition_stats = self.create_condition_specific_plots(all_points, use_weighted=True)
        weighted_combined_stats = self.create_combined_plot(all_points, use_weighted=True)
        
        # Print summary statistics
        print(f"\n📊 ANALYSIS RESULTS:")
        print(f"Total data points: {len(all_points)}")
        
        print(f"\n🔸 UNWEIGHTED Information Gain Results:")
        for condition, stats in unweighted_condition_stats.items():
            filtered_info = f" (filtered {stats['filtered_points']})" if stats.get('filtered_points', 0) > 0 else ""
            print(f"  {condition} condition{'s' if condition != 1 else ''}: "
                  f"R² = {stats['r_squared']:.3f}, "
                  f"Correlation = {stats['correlation']:.3f}, "
                  f"Points = {stats['n_points']}{filtered_info}")
        
        if unweighted_combined_stats and unweighted_combined_stats['correlation'] is not None:
            filtered_info = f" (filtered {unweighted_combined_stats['filtered_points']})" if unweighted_combined_stats.get('filtered_points', 0) > 0 else ""
            print(f"  Combined: R² = {unweighted_combined_stats['r_squared']:.3f}, "
                  f"Correlation = {unweighted_combined_stats['correlation']:.3f}, "
                  f"Points = {unweighted_combined_stats['n_points']}{filtered_info}")
        
        print(f"\n🔹 WEIGHTED Information Gain Results:")
        for condition, stats in weighted_condition_stats.items():
            filtered_info = f" (filtered {stats['filtered_points']})" if stats.get('filtered_points', 0) > 0 else ""
            print(f"  {condition} condition{'s' if condition != 1 else ''}: "
                  f"R² = {stats['r_squared']:.3f}, "
                  f"Correlation = {stats['correlation']:.3f}, "
                  f"Points = {stats['n_points']}{filtered_info}")
        
        if weighted_combined_stats and weighted_combined_stats['correlation'] is not None:
            filtered_info = f" (filtered {weighted_combined_stats['filtered_points']})" if weighted_combined_stats.get('filtered_points', 0) > 0 else ""
            print(f"  Combined: R² = {weighted_combined_stats['r_squared']:.3f}, "
                  f"Correlation = {weighted_combined_stats['correlation']:.3f}, "
                  f"Points = {weighted_combined_stats['n_points']}{filtered_info}")
        
        print(f"\nAll plots saved to directory: {self.output_dir}")
        print(f"  📁 Unweighted plots: {os.path.join(self.output_dir, 'unweighted')}")
        print(f"  📁 Weighted plots: {os.path.join(self.output_dir, 'weighted')}")
        
        # List created files
        for plot_type in ['unweighted', 'weighted']:
            subdir = os.path.join(self.output_dir, plot_type)
            if os.path.exists(subdir):
                plot_files = [f for f in os.listdir(subdir) if f.endswith('.png')]
                print(f"\n{plot_type.title()} plots ({len(plot_files)} files):")
                for plot_file in sorted(plot_files):
                    print(f"  - {plot_file}")

def main():
    # Configuration
    JSON_FILE = "/Users/ruoxiliu/Desktop/Argo_research/reasoning/sense/backtracking/gss2024_mapping_relevant_executed.json"  # Replace with your JSON file path
    CSV_FILE = "gss_2022.csv"     # Replace with your CSV file path  
    OUTPUT_DIR = "./info_gain_plots_relevant_1"            # Directory for output plots
    
    # Validate input files
    for file_path in [JSON_FILE, CSV_FILE]:
        if not os.path.exists(file_path):
            print(f"❌ File not found: {file_path}")
            print("Please update the file paths in the main() function.")
            return
    
    # Create plotter and run analysis
    plotter = InformationGainKLPlotter(JSON_FILE, CSV_FILE, OUTPUT_DIR)
    
    try:
        plotter.run_analysis()
        print("\n🎉 ANALYSIS COMPLETE!")
        
    except Exception as e:
        print(f"\n❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()