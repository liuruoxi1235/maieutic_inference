import json
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import os

class RandVsRelevantComparator:
    def __init__(self, rand_file, relevant_file, output_dir):
        self.rand_file = rand_file
        self.relevant_file = relevant_file
        self.output_dir = output_dir
        
    def load_data(self):
        """Load both JSON files"""
        print(f"Loading random constraints data from: {self.rand_file}")
        with open(self.rand_file, 'r') as f:
            self.rand_data = json.load(f)
        
        print(f"Loading relevant constraints data from: {self.relevant_file}")
        with open(self.relevant_file, 'r') as f:
            self.relevant_data = json.load(f)
        
        print(f"✓ Loaded {len(self.rand_data)} variables from random file")
        print(f"✓ Loaded {len(self.relevant_data)} variables from relevant file")
    
    def extract_variable_data(self):
        """Extract variables that exist in both datasets with valid prompts"""
        common_vars = {}
        
        # Iterate over variables in rand data
        for var_name, rand_var_data in self.rand_data.items():
            # Check if variable exists in relevant data
            if var_name not in self.relevant_data:
                continue
            
            relevant_var_data = self.relevant_data[var_name]
            
            # Check if both have valid prompts
            rand_has_prompts = ('rand_prompts' in rand_var_data and 
                               rand_var_data['rand_prompts'] is not None and 
                               len(rand_var_data['rand_prompts']) > 0)
            
            relevant_has_prompts = ('relevant_prompts' in relevant_var_data and 
                                   relevant_var_data['relevant_prompts'] is not None and 
                                   len(relevant_var_data['relevant_prompts']) > 0)
            
            if not (rand_has_prompts and relevant_has_prompts):
                continue
            
            # Extract KL data by constraint size
            rand_kl_by_constraint = {}
            relevant_kl_by_constraint = {}
            
            # Extract from rand prompts
            for prompt in rand_var_data['rand_prompts']:
                if 'KL' in prompt and 'num_conditions' in prompt:
                    num_cond = prompt['num_conditions']
                    kl_val = prompt['KL']
                    rand_kl_by_constraint[num_cond] = kl_val
            
            # Extract from relevant prompts
            for prompt in relevant_var_data['relevant_prompts']:
                if 'KL' in prompt and 'num_conditions' in prompt:
                    num_cond = prompt['num_conditions']
                    kl_val = prompt['KL']
                    relevant_kl_by_constraint[num_cond] = kl_val
            
            # Only include if we have KL data for both
            if rand_kl_by_constraint and relevant_kl_by_constraint:
                # Get average KL for coloring (prefer from rand data, fallback to relevant)
                avg_kl = (rand_var_data.get('avg_KL') or 
                         relevant_var_data.get('avg_KL') or 0)
                
                common_vars[var_name] = {
                    'rand_kl': rand_kl_by_constraint,
                    'relevant_kl': relevant_kl_by_constraint,
                    'avg_kl': avg_kl
                }
        
        print(f"Found {len(common_vars)} variables with valid prompts in both datasets")
        return common_vars
    
    def create_constraint_specific_plots(self, common_vars):
        """Create plots for each constraint size (1-5)"""
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        constraint_stats = {}
        all_points = []  # For the combined plot
        
        # Find all constraint sizes present in the data
        all_constraints = set()
        for var_data in common_vars.values():
            all_constraints.update(var_data['rand_kl'].keys())
            all_constraints.update(var_data['relevant_kl'].keys())
        all_constraints = sorted(list(all_constraints))
        
        print(f"Found constraint sizes: {all_constraints}")
        
        # Create plot for each constraint size
        for constraint_size in all_constraints:
            relevant_vals = []
            rand_vals = []
            colors = []
            var_names = []
            filtered_count = 0
            
            # Collect data points for this constraint size
            for var_name, var_data in common_vars.items():
                if (constraint_size in var_data['rand_kl'] and 
                    constraint_size in var_data['relevant_kl']):
                    
                    relevant_kl = var_data['relevant_kl'][constraint_size]
                    rand_kl = var_data['rand_kl'][constraint_size]
                    avg_kl = var_data['avg_kl']
                    
                    # Filter out points with KL > 2
                    if relevant_kl > 2 or rand_kl > 2:
                        filtered_count += 1
                        continue
                    
                    relevant_vals.append(relevant_kl)
                    rand_vals.append(rand_kl)
                    colors.append(avg_kl)
                    var_names.append(var_name)
                    
                    # Store for combined plot
                    all_points.append({
                        'relevant': relevant_kl,
                        'rand': rand_kl,
                        'color': avg_kl,
                        'constraint': constraint_size,
                        'var_name': var_name
                    })
            
            if filtered_count > 0:
                print(f"  Filtered out {filtered_count} points with KL > 2 for constraint {constraint_size}")
            
            if len(relevant_vals) < 2:
                print(f"Skipping constraint {constraint_size}: insufficient data points ({len(relevant_vals)})")
                continue
            
            # Create the plot
            plt.figure(figsize=(10, 8))
            
            # Create scatter plot with color mapping
            scatter = plt.scatter(relevant_vals, rand_vals, c=colors, alpha=0.7, s=80, cmap='viridis')
            
            # Add colorbar
            cbar = plt.colorbar(scatter)
            cbar.set_label('Variable Difficulty (avg KL)', fontsize=12)
            
            # Add diagonal line (y=x) for reference
            min_val = min(min(relevant_vals), min(rand_vals))
            max_val = max(max(relevant_vals), max(rand_vals))
            plt.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, linewidth=2,
                    label='y = x (equal performance)')
            
            # Calculate correlation and line of best fit
            if len(relevant_vals) > 1:
                correlation = np.corrcoef(relevant_vals, rand_vals)[0, 1]
                slope, intercept, r_value, p_value, std_err = stats.linregress(relevant_vals, rand_vals)
                
                # Plot line of best fit
                x_line = np.linspace(min(relevant_vals), max(relevant_vals), 100)
                y_line = slope * x_line + intercept
                plt.plot(x_line, y_line, 'r-', alpha=0.8, linewidth=2,
                        label=f'Best fit (R² = {r_value**2:.3f})')
                
                constraint_stats[constraint_size] = {
                    'correlation': correlation,
                    'r_squared': r_value**2,
                    'slope': slope,
                    'intercept': intercept,
                    'n_points': len(relevant_vals),
                    'filtered_points': filtered_count
                }
            
            plt.xlabel('KL Divergence (Relevant Constraints)', fontsize=12)
            plt.ylabel('KL Divergence (Random Constraints)', fontsize=12)
            plt.title(f'Random vs Relevant Constraints: {constraint_size} Condition{"s" if constraint_size != 1 else ""}\n(Colored by Variable Difficulty, KL ≤ 2)', fontsize=14)
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Add statistics text
            stats_text = f'Points: {len(relevant_vals)}'
            if filtered_count > 0:
                stats_text += f' (filtered: {filtered_count})'
            if len(relevant_vals) > 1:
                stats_text += f'\nCorrelation: {correlation:.3f}'
            if colors:
                stats_text += f'\nAvg difficulty: {np.mean(colors):.3f}'
                
            plt.text(0.02, 0.98, stats_text,
                    transform=plt.gca().transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            # Save the plot
            output_path = os.path.join(self.output_dir, f'rand_vs_relevant_{constraint_size}_constraints.png')
            plt.tight_layout()
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Created plot for {constraint_size} constraint{'s' if constraint_size != 1 else ''}: {len(relevant_vals)} points")
        
        return constraint_stats, all_points
    
    def create_combined_plot(self, all_points):
        """Create a combined plot with all constraint sizes"""
        if not all_points:
            print("No data points for combined plot")
            return
        
        # Filter out points with KL > 2 and count filtered points
        filtered_points = [p for p in all_points if p['relevant'] <= 2 and p['rand'] <= 2]
        filtered_out_count = len(all_points) - len(filtered_points)
        
        if filtered_out_count > 0:
            print(f"  Filtered out {filtered_out_count} points with KL > 2 for combined plot")
        
        if not filtered_points:
            print("No data points remaining after filtering for combined plot")
            return
        
        # Extract data for plotting
        relevant_vals = [p['relevant'] for p in filtered_points]
        rand_vals = [p['rand'] for p in filtered_points]
        colors = [p['color'] for p in filtered_points]
        
        plt.figure(figsize=(12, 10))
        
        # Create scatter plot with color mapping
        scatter = plt.scatter(relevant_vals, rand_vals, c=colors, alpha=0.6, s=60, cmap='viridis')
        
        # Add colorbar
        cbar = plt.colorbar(scatter)
        cbar.set_label('Variable Difficulty (avg KL)', fontsize=12)
        
        # Add diagonal line (y=x) for reference
        min_val = min(min(relevant_vals), min(rand_vals))
        max_val = max(max(relevant_vals), max(rand_vals))
        plt.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, linewidth=2,
                label='y = x (equal performance)')
        
        # Calculate overall correlation and line of best fit
        if len(relevant_vals) > 1:
            correlation = np.corrcoef(relevant_vals, rand_vals)[0, 1]
            slope, intercept, r_value, p_value, std_err = stats.linregress(relevant_vals, rand_vals)
            
            # Plot line of best fit
            x_line = np.linspace(min(relevant_vals), max(relevant_vals), 100)
            y_line = slope * x_line + intercept
            plt.plot(x_line, y_line, 'r-', alpha=0.8, linewidth=2,
                    label=f'Best fit (R² = {r_value**2:.3f})')
        
        plt.xlabel('KL Divergence (Relevant Constraints)', fontsize=12)
        plt.ylabel('KL Divergence (Random Constraints)', fontsize=12)
        plt.title('Random vs Relevant Constraints: All Constraint Sizes Combined\n(Colored by Variable Difficulty, KL ≤ 2)', fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Add statistics text
        stats_text = f'Points: {len(relevant_vals)}'
        if filtered_out_count > 0:
            stats_text += f' (filtered: {filtered_out_count})'
        if len(relevant_vals) > 1:
            stats_text += f'\nCorrelation: {correlation:.3f}'
        if colors:
            stats_text += f'\nAvg difficulty: {np.mean(colors):.3f}'
        
        # Count points by constraint size
        constraint_counts = {}
        for point in filtered_points:
            constraint = point['constraint']
            constraint_counts[constraint] = constraint_counts.get(constraint, 0) + 1
        
        constraint_breakdown = ', '.join([f'{k}: {v}' for k, v in sorted(constraint_counts.items())])
        stats_text += f'\nBy constraint: {constraint_breakdown}'
            
        plt.text(0.02, 0.98, stats_text,
                transform=plt.gca().transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Save the plot
        output_path = os.path.join(self.output_dir, 'rand_vs_relevant_combined.png')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Created combined plot with {len(relevant_vals)} total points")
        
        return {
            'correlation': correlation if len(relevant_vals) > 1 else None,
            'r_squared': r_value**2 if len(relevant_vals) > 1 else None,
            'slope': slope if len(relevant_vals) > 1 else None,
            'intercept': intercept if len(relevant_vals) > 1 else None,
            'n_points': len(relevant_vals),
            'filtered_points': filtered_out_count
        }
    
    def run_comparison(self):
        """Run the full comparison analysis"""
        print("Random vs Relevant Constraints Comparison")
        print("=" * 50)
        
        # Load data
        self.load_data()
        
        # Extract common variables
        common_vars = self.extract_variable_data()
        
        if not common_vars:
            print("❌ No common variables found with valid prompts in both datasets")
            return
        
        # Create constraint-specific plots
        print(f"\nCreating constraint-specific plots...")
        constraint_stats, all_points = self.create_constraint_specific_plots(common_vars)
        
        # Create combined plot
        print(f"\nCreating combined plot...")
        combined_stats = self.create_combined_plot(all_points)
        
        # Print summary statistics
        print(f"\n📊 COMPARISON RESULTS:")
        print(f"Variables analyzed: {len(common_vars)}")
        print(f"Total data points: {len(all_points)}")
        
        print(f"\nConstraint-specific Results:")
        for constraint, stats in constraint_stats.items():
            filtered_info = f" (filtered {stats['filtered_points']})" if stats.get('filtered_points', 0) > 0 else ""
            print(f"  {constraint} constraint{'s' if constraint != 1 else ''}: "
                  f"R² = {stats['r_squared']:.3f}, "
                  f"Correlation = {stats['correlation']:.3f}, "
                  f"Points = {stats['n_points']}{filtered_info}")
        
        if combined_stats and combined_stats['correlation'] is not None:
            filtered_info = f" (filtered {combined_stats['filtered_points']})" if combined_stats.get('filtered_points', 0) > 0 else ""
            print(f"\nCombined Results:")
            print(f"  Overall correlation: {combined_stats['correlation']:.3f}")
            print(f"  Overall R²: {combined_stats['r_squared']:.3f}")
            print(f"  Total points: {combined_stats['n_points']}{filtered_info}")
        
        print(f"\nAll plots saved to directory: {self.output_dir}")
        
        # List created files
        plot_files = [f for f in os.listdir(self.output_dir) if f.endswith('.png')]
        print(f"Created {len(plot_files)} plot files:")
        for plot_file in sorted(plot_files):
            print(f"  - {plot_file}")

def main():
    # Configuration
    RAND_FILE = "/Users/ruoxiliu/Desktop/Argo_research/reasoning/sense/backtracking/gss2024_mapping_rand_executed.json"  # Replace with your random constraints file
    RELEVANT_FILE = "/Users/ruoxiliu/Desktop/Argo_research/reasoning/sense/backtracking/gss2024_mapping_relevant_executed.json"  # Replace with your relevant constraints file
    OUTPUT_DIR = "./plot_randrel_compare_less2"  # Directory for comparison plots
    
    # Validate input files
    for file_path in [RAND_FILE, RELEVANT_FILE]:
        if not os.path.exists(file_path):
            print(f"❌ File not found: {file_path}")
            print("Please update the file paths in the main() function.")
            return
    
    # Create comparator and run analysis
    comparator = RandVsRelevantComparator(RAND_FILE, RELEVANT_FILE, OUTPUT_DIR)
    
    try:
        comparator.run_comparison()
        print("\n🎉 COMPARISON COMPLETE!")
        
    except Exception as e:
        print(f"\n❌ Error during comparison: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()