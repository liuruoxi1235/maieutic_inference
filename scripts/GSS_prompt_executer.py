import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import entropy
from scipy import stats
import re
from openai import OpenAI
import ast

class LLMPromptProcessor:
    def __init__(self, api_key, model_id="gpt-4", temperature=0.7, log_path="llm_log.txt"):
        self.client = OpenAI(api_key=api_key)
        self.model_ID = model_id
        self.temperature = temperature
        self.log_pth = log_path
        self.LLM_budget_left = 1000  # Adjust as needed
    
    def single_LLM_call(self, json_prompt, client, name, replacements={}, additional_message=None):
        """
        Take a json styled role-name array and return the generated content, supports placeholder replacements
        """
        filled_prompt = []
        
        for role, content in json_prompt:
            # Perform all replacements on the each content string
            for placeholder, value in replacements.items():
                content = content.replace(f"{{{placeholder}}}", str(value))
            filled_prompt.append((role, content))
        
        # Build the final messages list
        messages = [{"role": role, "content": content} for role, content in filled_prompt]

        if additional_message is not None:
            if isinstance(additional_message, list):
                messages.extend([{"role": role, "content": content} for role, content in additional_message])
            else:
                with open(self.log_pth, "a") as log_file:
                    log_file.write("[err] additional_message must be a list of exactly two (role, content) tuples.\n\n")
        
        with open(self.log_pth, "a") as log_file:
            log_file.write(f"[main {name}] [prompt] : {messages}\n")

        # Call the model
        response = client.chat.completions.create(
            model=self.model_ID,
            messages=messages,
            temperature=self.temperature
        )
        
        self.LLM_budget_left -= 1

        # Return the actual generation
        return response.choices[0].message.content
    
    def parse_prediction(self, prediction_text):
        """
        Parse the prediction text to extract the probability distribution
        """
        # Look for patterns like [0.1234, 0.5678, 0.2088] or [0.1234; 0.5678; 0.2088]
        pattern = r'\[([\d\.\;\,\s]+)\]'
        match = re.search(pattern, prediction_text)
        
        if match:
            # Extract the numbers
            numbers_str = match.group(1)
            # Replace semicolons with commas if present
            numbers_str = numbers_str.replace(';', ',')
            
            try:
                # Try to parse as a list
                numbers = [float(x.strip()) for x in numbers_str.split(',')]
                return np.array(numbers)
            except:
                # If that fails, try to evaluate as Python expression
                try:
                    return np.array(ast.literal_eval(f"[{numbers_str}]"))
                except:
                    return None
        return None
    
    def compute_kl_divergence(self, p, q, epsilon=1e-10):
        """
        Compute KL divergence between two probability distributions
        KL(P||Q) = sum(P * log(P/Q))
        """
        # Ensure distributions are normalized
        p = np.array(p)
        q = np.array(q)
        
        # Add small epsilon to avoid log(0)
        p = p + epsilon
        q = q + epsilon
        
        # Normalize
        p = p / np.sum(p)
        q = q / np.sum(q)
        
        # Compute KL divergence
        return entropy(p, q)
    
    def process_variable_mapping(self, input_file_path, output_file_path, max_var=None):
        """
        Process the variable mapping file and compute predictions and KL divergences
        """
        with open(input_file_path, 'r') as f:
            data = json.load(f)
        
        processed_vars = 0
        all_kl_values = []
        all_num_conditions = []
        
        for var_name, var_data in data.items():
            if max_var is not None and processed_vars >= max_var:
                break
            
            # Determine which prompts field to use
            has_rand = 'rand_prompts' in var_data and var_data['rand_prompts'] is not None
            has_relevant = 'relevant_prompts' in var_data and var_data['relevant_prompts'] is not None
            
            # Skip if both or neither are present
            if (has_rand and has_relevant) or (not has_rand and not has_relevant):
                continue
            
            # Get the appropriate prompts field
            prompts_field = 'rand_prompts' if has_rand else 'relevant_prompts'
            prompts_data = var_data[prompts_field]
            
            # Check if all prompts for this variable already have predictions
            all_have_predictions = all(
                'prediction' in prompt and 'KL' in prompt 
                for prompt in prompts_data
            )
            
            if all_have_predictions:
                # Still collect KL values for plotting
                for prompt in prompts_data:
                    all_kl_values.append(prompt['KL'])
                    all_num_conditions.append(prompt['num_conditions'])
                continue
            
            print(f"Processing variable: {var_name} (using {prompts_field})")
            
            for i, prompt_data in enumerate(prompts_data):
                # Skip if already has prediction and KL
                if 'prediction' in prompt_data and 'KL' in prompt_data:
                    all_kl_values.append(prompt_data['KL'])
                    all_num_conditions.append(prompt_data['num_conditions'])
                    continue
                
                # Make LLM call
                try:
                    prediction_text = self.single_LLM_call(
                        prompt_data['prompt'], 
                        self.client, 
                        f"{var_name}_{i}"
                    )
                    
                    # Parse prediction
                    prediction_array = self.parse_prediction(prediction_text)
                    
                    if prediction_array is not None:
                        # Parse completion
                        completion_text = prompt_data['completion']
                        completion_array = self.parse_prediction(completion_text)
                        
                        if completion_array is not None and len(prediction_array) == len(completion_array):
                            # Compute KL divergence
                            kl_value = self.compute_kl_divergence(completion_array, prediction_array)
                            
                            # Store results
                            prompt_data['prediction'] = prediction_text
                            prompt_data['KL'] = float(kl_value)
                            
                            all_kl_values.append(kl_value)
                            all_num_conditions.append(prompt_data['num_conditions'])
                            
                            print(f"  Prompt {i}: KL = {kl_value:.4f}")
                        else:
                            print(f"  Prompt {i}: Failed to parse arrays or dimension mismatch")
                    else:
                        print(f"  Prompt {i}: Failed to parse prediction")
                        
                except Exception as e:
                    print(f"  Prompt {i}: Error - {str(e)}")
            
            processed_vars += 1
        
        # Compute average KL for each variable
        for var_name, var_data in data.items():
            # Determine which prompts field exists
            has_rand = 'rand_prompts' in var_data and var_data['rand_prompts'] is not None
            has_relevant = 'relevant_prompts' in var_data and var_data['relevant_prompts'] is not None
            
            # Skip if both or neither are present
            if (has_rand and has_relevant) or (not has_rand and not has_relevant):
                continue
            
            prompts_field = 'rand_prompts' if has_rand else 'relevant_prompts'
            prompts_data = var_data[prompts_field]
                
            kl_values = []
            for prompt in prompts_data:
                if 'KL' in prompt:
                    kl_values.append(prompt['KL'])
            
            if kl_values:
                var_data['avg_KL'] = float(np.mean(kl_values))
                print(f"Variable {var_name}: avg_KL = {var_data['avg_KL']:.4f}")
        
        # Save updated data to output file
        with open(output_file_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        return all_num_conditions, all_kl_values, data
    
    def create_scatter_plot(self, num_conditions, kl_values, data, output_path="kl_scatter_plot.png", kl_threshold=None):
        """
        Create scatter plot with line of best fit, colored by variable difficulty
        """
        # Create mapping from (num_conditions, kl_value) to variable avg_KL
        var_avg_kls = {}
        point_colors = []
        
        # Build mapping of KL values to variables and their average KL
        for var_name, var_data in data.items():
            # Check for both types of prompts
            has_rand = 'rand_prompts' in var_data and var_data['rand_prompts'] is not None
            has_relevant = 'relevant_prompts' in var_data and var_data['relevant_prompts'] is not None
            
            # Skip if both or neither are present
            if (has_rand and has_relevant) or (not has_rand and not has_relevant):
                continue
            
            if 'avg_KL' not in var_data:
                continue
            
            avg_kl = var_data['avg_KL']
            prompts_field = 'rand_prompts' if has_rand else 'relevant_prompts'
            
            for prompt in var_data[prompts_field]:
                if 'KL' in prompt and 'num_conditions' in prompt:
                    key = (prompt['num_conditions'], prompt['KL'])
                    var_avg_kls[key] = avg_kl
        
        # Filter data if threshold is specified
        if kl_threshold is not None:
            filtered_indices = [i for i, kl in enumerate(kl_values) if kl < kl_threshold]
            filtered_num_conditions = [num_conditions[i] for i in filtered_indices]
            filtered_kl_values = [kl_values[i] for i in filtered_indices]
            filtered_out_count = len(kl_values) - len(filtered_kl_values)
        else:
            filtered_num_conditions = num_conditions
            filtered_kl_values = kl_values
            filtered_out_count = 0
        
        # Get colors for each point (after filtering)
        for nc, kl in zip(filtered_num_conditions, filtered_kl_values):
            key = (nc, kl)
            if key in var_avg_kls:
                point_colors.append(var_avg_kls[key])
            else:
                point_colors.append(np.mean(list(var_avg_kls.values())) if var_avg_kls else 0)
        
        plt.figure(figsize=(12, 8))
        
        # Create scatter plot with color mapping
        scatter = plt.scatter(filtered_num_conditions, filtered_kl_values, c=point_colors, 
                            alpha=0.7, s=60, cmap='viridis')
        
        # Add colorbar
        cbar = plt.colorbar(scatter)
        cbar.set_label('Variable Difficulty (avg KL)', fontsize=12)
        
        # Calculate line of best fit
        if len(filtered_num_conditions) > 1:
            slope, intercept, r_value, p_value, std_err = stats.linregress(filtered_num_conditions, filtered_kl_values)
            
            # Create line of best fit
            x_line = np.linspace(min(filtered_num_conditions), max(filtered_num_conditions), 100)
            y_line = slope * x_line + intercept
            
            plt.plot(x_line, y_line, 'r-', alpha=0.8, linewidth=2,
                    label=f'Line of best fit (R² = {r_value**2:.3f})')
        
        plt.xlabel('Number of Conditions', fontsize=12)
        plt.ylabel('KL Divergence', fontsize=12)
        
        # Set title based on whether filtering was applied
        if kl_threshold is not None:
            plt.title(f'KL Divergence vs Number of Conditions (KL < {kl_threshold})\n(Colored by Variable Difficulty)', fontsize=14)
        else:
            plt.title('KL Divergence vs Number of Conditions\n(Colored by Variable Difficulty)', fontsize=14)
        
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Add some statistics
        stats_text = f'Points: {len(filtered_num_conditions)}'
        if filtered_out_count > 0:
            stats_text += f' (filtered: {filtered_out_count})'
        if point_colors:
            stats_text += f'\nAvg difficulty: {np.mean(point_colors):.3f}'
        
        plt.text(0.02, 0.98, stats_text,
                transform=plt.gca().transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()  # Close figure to free memory
        
        if len(filtered_num_conditions) > 1:
            return slope, intercept, r_value**2, filtered_out_count
        else:
            return None, None, None, filtered_out_count
    
    def create_pairwise_comparison_plots(self, data, output_dir):
        """
        Create pairwise comparison plots for different numbers of conditions
        """
        import os
        from itertools import combinations
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Extract KL values organized by variable and number of conditions
        var_kl_data = {}
        var_avg_kls = {}
        
        for var_name, var_data in data.items():
            # Check for both types of prompts
            has_rand = 'rand_prompts' in var_data and var_data['rand_prompts'] is not None
            has_relevant = 'relevant_prompts' in var_data and var_data['relevant_prompts'] is not None
            
            # Skip if both or neither are present
            if (has_rand and has_relevant) or (not has_rand and not has_relevant):
                continue
                
            var_kl_data[var_name] = {}
            
            # Get average KL for coloring
            if 'avg_KL' in var_data:
                var_avg_kls[var_name] = var_data['avg_KL']
            else:
                var_avg_kls[var_name] = 0
            
            prompts_field = 'rand_prompts' if has_rand else 'relevant_prompts'
            
            for prompt in var_data[prompts_field]:
                if 'KL' in prompt and 'num_conditions' in prompt:
                    num_cond = prompt['num_conditions']
                    kl_val = prompt['KL']
                    var_kl_data[var_name][num_cond] = kl_val
        
        # Find all unique condition numbers
        all_conditions = set()
        for var_data in var_kl_data.values():
            all_conditions.update(var_data.keys())
        all_conditions = sorted(list(all_conditions))
        
        print(f"Found conditions: {all_conditions}")
        
        # Create pairwise comparison plots
        comparison_stats = {}
        
        for i, cond_a in enumerate(all_conditions):
            for j, cond_b in enumerate(all_conditions):
                if i >= j:  # Only create plots for unique pairs where cond_a < cond_b
                    continue
                
                # Collect data points for this comparison
                x_values = []  # KL values for smaller condition number
                y_values = []  # KL values for larger condition number
                colors = []   # Variable difficulty (avg_KL) for coloring
                var_names = []
                
                for var_name, kl_dict in var_kl_data.items():
                    if cond_a in kl_dict and cond_b in kl_dict:
                        x_values.append(kl_dict[cond_a])
                        y_values.append(kl_dict[cond_b])
                        colors.append(var_avg_kls.get(var_name, 0))
                        var_names.append(var_name)
                
                if len(x_values) < 2:
                    print(f"Skipping comparison {cond_a} vs {cond_b}: insufficient data points ({len(x_values)})")
                    continue
                
                # Count variables where each side is better (lower KL)
                x_better_count = sum(1 for x, y in zip(x_values, y_values) if x < y)  # fewer conditions better
                y_better_count = sum(1 for x, y in zip(x_values, y_values) if y < x)  # more conditions better
                equal_count = sum(1 for x, y in zip(x_values, y_values) if x == y)
                
                # Calculate ratio (more conditions better / fewer conditions better)
                if x_better_count > 0:
                    better_ratio = y_better_count / x_better_count
                else:
                    better_ratio = float('inf') if y_better_count > 0 else 0
                
                # Create the plot
                plt.figure(figsize=(12, 8))
                
                # Create scatter plot with color mapping
                scatter = plt.scatter(x_values, y_values, c=colors, alpha=0.7, s=80, cmap='viridis')
                
                # Add colorbar
                cbar = plt.colorbar(scatter)
                cbar.set_label('Variable Difficulty (avg KL)', fontsize=12)
                
                # Add diagonal line (y=x) for reference
                min_val = min(min(x_values), min(y_values))
                max_val = max(max(x_values), max(y_values))
                plt.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, linewidth=2,
                        label='y = x (equal performance)')
                
                # Calculate correlation and line of best fit
                if len(x_values) > 1:
                    correlation = np.corrcoef(x_values, y_values)[0, 1]
                    slope, intercept, r_value, p_value, std_err = stats.linregress(x_values, y_values)
                    
                    # Plot line of best fit
                    x_line = np.linspace(min(x_values), max(x_values), 100)
                    y_line = slope * x_line + intercept
                    plt.plot(x_line, y_line, 'r-', alpha=0.8, linewidth=2,
                            label=f'Best fit (R² = {r_value**2:.3f})')
                    
                    comparison_stats[f"{cond_a}_vs_{cond_b}"] = {
                        'correlation': correlation,
                        'r_squared': r_value**2,
                        'slope': slope,
                        'intercept': intercept,
                        'n_points': len(x_values),
                        'fewer_cond_better': x_better_count,
                        'more_cond_better': y_better_count,
                        'equal': equal_count,
                        'ratio': better_ratio
                    }
                
                plt.xlabel(f'KL Divergence ({cond_a} condition{"s" if cond_a != 1 else ""})', fontsize=12)
                plt.ylabel(f'KL Divergence ({cond_b} condition{"s" if cond_b != 1 else ""})', fontsize=12)
                plt.title(f'KL Divergence Comparison: {cond_a} vs {cond_b} Conditions\n(Colored by Variable Difficulty)', fontsize=14)
                plt.legend()
                plt.grid(True, alpha=0.3)
                
                # Add statistics text with performance comparison
                stats_text = f'Points: {len(x_values)}'
                if len(x_values) > 1:
                    stats_text += f'\nCorrelation: {correlation:.3f}'
                if colors:
                    stats_text += f'\nAvg difficulty: {np.mean(colors):.3f}'
                
                # Add performance comparison
                stats_text += f'\n{cond_a} cond better: {x_better_count}'
                stats_text += f'\n{cond_b} cond better: {y_better_count}'
                if equal_count > 0:
                    stats_text += f'\nEqual: {equal_count}'
                
                # Add ratio
                if better_ratio != float('inf') and better_ratio != 0:
                    stats_text += f'\nRatio: {better_ratio:.2f}'
                elif better_ratio == float('inf'):
                    stats_text += f'\nRatio: ∞'
                else:
                    stats_text += f'\nRatio: 0'
                    
                plt.text(0.02, 0.98, stats_text,
                        transform=plt.gca().transAxes, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                
                # Save the plot
                output_path = os.path.join(output_dir, f'comparison_{cond_a}_vs_{cond_b}.png')
                plt.tight_layout()
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                plt.close()
                
                print(f"Created comparison plot: {cond_a} vs {cond_b} conditions ({len(x_values)} points)")
                print(f"  {cond_a} cond better: {x_better_count}, {cond_b} cond better: {y_better_count}, ratio: {better_ratio:.2f}")
        
        return comparison_stats

def main():
    # Configuration
    API_KEY = 'YOUR_OPENAI_API_KEY'  # Replace with your actual API key
    INPUT_FILE = "/Users/ruoxiliu/Desktop/Argo_research/reasoning/sense/backtracking/gss2024_mapping_rand_executed.json"  # Replace with your input file path
    OUTPUT_FILE = "/Users/ruoxiliu/Desktop/Argo_research/reasoning/sense/backtracking/gss2024_mapping_rand_temp.json"  # Output file for processed data
    OUTPUT_DIR = "./plots_rand"  # Directory for all plots
    MAX_VAR = 100  # Set to None to process all variables
    
    # Initialize processor
    processor = LLMPromptProcessor(API_KEY)
    
    # Process the variable mapping file
    print("Processing variable mapping file...")
    num_conditions, kl_values, processed_data = processor.process_variable_mapping(
        INPUT_FILE, 
        OUTPUT_FILE,
        max_var=MAX_VAR
    )
    
    print(f"Saved processed data to: {OUTPUT_FILE}")
    
    if len(num_conditions) > 0:
        import os
        
        # Create output directory
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        # Create overall scatter plot (all data)
        print(f"\nCreating overall scatter plot with {len(num_conditions)} data points...")
        overall_plot_path = os.path.join(OUTPUT_DIR, "overall_kl_vs_conditions.png")
        slope, intercept, r_squared, filtered_count = processor.create_scatter_plot(
            num_conditions, kl_values, processed_data, overall_plot_path
        )
        
        print(f"Overall plot saved to: {overall_plot_path}")
        if slope is not None:
            print(f"Overall Results:")
            print(f"  Slope: {slope:.4f}")
            print(f"  Intercept: {intercept:.4f}")
            print(f"  R-squared: {r_squared:.4f}")
        
        # Create filtered scatter plot (KL < 1)
        print(f"\nCreating filtered scatter plot (KL < 1)...")
        filtered_plot_path = os.path.join(OUTPUT_DIR, "overall_kl_vs_conditions_filtered.png")
        slope_filtered, intercept_filtered, r_squared_filtered, filtered_count = processor.create_scatter_plot(
            num_conditions, kl_values, processed_data, filtered_plot_path, kl_threshold=1.0
        )
        
        print(f"Filtered plot saved to: {filtered_plot_path}")
        if slope_filtered is not None:
            print(f"Filtered Results (KL < 1):")
            print(f"  Slope: {slope_filtered:.4f}")
            print(f"  Intercept: {intercept_filtered:.4f}")
            print(f"  R-squared: {r_squared_filtered:.4f}")
            print(f"  Filtered out: {filtered_count} points")
        else:
            print(f"  Insufficient data points after filtering (filtered out: {filtered_count} points)")
        
        # Create pairwise comparison plots
        print(f"\nCreating pairwise comparison plots...")
        comparison_stats = processor.create_pairwise_comparison_plots(processed_data, OUTPUT_DIR)
        
        print(f"\nPairwise Comparison Results:")
        for comparison, stats in comparison_stats.items():
            parts = comparison.split('_vs_')
            cond_a, cond_b = parts[0], parts[1]
            print(f"  {comparison.replace('_', ' ')}: R² = {stats['r_squared']:.3f}, "
                  f"Correlation = {stats['correlation']:.3f}, Points = {stats['n_points']}")
            print(f"    {cond_a} cond better: {stats['fewer_cond_better']}, "
                  f"{cond_b} cond better: {stats['more_cond_better']}, "
                  f"ratio: {stats['ratio']:.2f}")
            if stats.get('equal', 0) > 0:
                print(f"    Equal: {stats['equal']}")
        
        # Basic statistics
        print(f"\nKL Divergence Statistics:")
        print(f"  Mean: {np.mean(kl_values):.4f}")
        print(f"  Std: {np.std(kl_values):.4f}")
        print(f"  Min: {np.min(kl_values):.4f}")
        print(f"  Max: {np.max(kl_values):.4f}")
        
        # Statistics for filtered data
        filtered_kl_values = [kl for kl in kl_values if kl < 1.0]
        if filtered_kl_values:
            print(f"\nFiltered KL Divergence Statistics (KL < 1):")
            print(f"  Count: {len(filtered_kl_values)} / {len(kl_values)}")
            print(f"  Mean: {np.mean(filtered_kl_values):.4f}")
            print(f"  Std: {np.std(filtered_kl_values):.4f}")
            print(f"  Min: {np.min(filtered_kl_values):.4f}")
            print(f"  Max: {np.max(filtered_kl_values):.4f}")
        
        print(f"\nAll plots saved to directory: {OUTPUT_DIR}")
        
        # List all created files
        plot_files = [f for f in os.listdir(OUTPUT_DIR) if f.endswith('.png')]
        print(f"Created {len(plot_files)} plot files:")
        for plot_file in sorted(plot_files):
            print(f"  - {plot_file}")
            
    else:
        print("No data points found to plot.")

if __name__ == "__main__":
    main()