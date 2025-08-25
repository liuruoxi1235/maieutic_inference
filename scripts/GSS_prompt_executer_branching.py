import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import entropy
from scipy import stats
import re
from openai import OpenAI
import ast
import os
import pandas as pd

class BranchingPromptProcessor:
    def __init__(self, api_key, csv_file, model_id="gpt-4", temperature=0.7, log_path="branching_llm_log.txt"):
        self.client = OpenAI(api_key=api_key)
        self.model_ID = model_id
        self.temperature = temperature
        self.log_pth = log_path
        self.LLM_budget_left = 1000  # Adjust as needed
        
        # Load CSV data for information gain calculations
        print(f"Loading CSV data from: {csv_file}")
        self.df = pd.read_csv(csv_file, low_memory=False)
        print(f"✓ Loaded CSV with {len(self.df)} rows and {len(self.df.columns)} columns")
        
        # Define responses to exclude
        self.INVALID_RESPONSES = {"don't know", "iap", "not available in this year", "no answer", "skipped on web", "refused"}
    
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
            log_file.write(f"[branching {name}] [prompt] : {messages}\n")

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
        """Get distribution of variable given constraints"""
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
        initial_count = len(filtered_df)
        
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
            raise ValueError(f"Insufficient data: only {len(valid_series)} valid responses after applying constraints (started with {initial_count})")
        
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
                unique_categories = list(distribution.index)
                raise ValueError(f"Insufficient diversity: only {len(distribution)} unique categories: {unique_categories}")
        
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
    
    def process_branching_prompts(self, input_file_path, output_file_path, max_var=None):
        """
        Process branching prompts and compute predictions and KL divergences
        """
        with open(input_file_path, 'r') as f:
            data = json.load(f)
        
        processed_vars = 0
        all_kl_values = []
        all_info_gains = []
        all_weighted_info_gains = []
        all_num_original_conditions = []
        all_h_branching_original = []
        
        # First pass: collect ALL existing data points (from previous runs)
        print("Collecting existing data points from previous runs...")
        for var_name, var_data in data.items():
            # Determine which prompts field to use
            has_rand = 'rand_prompts' in var_data and var_data['rand_prompts'] is not None
            has_relevant = 'relevant_prompts' in var_data and var_data['relevant_prompts'] is not None
            
            # Skip if both or neither are present
            if (has_rand and has_relevant) or (not has_rand and not has_relevant):
                continue
            
            # Get the appropriate prompts field
            prompts_field = 'rand_prompts' if has_rand else 'relevant_prompts'
            prompts_data = var_data[prompts_field]
            
            for i, prompt_data in enumerate(prompts_data):
                # Skip if no branching prompt
                if 'branching_prompt' not in prompt_data or prompt_data['branching_prompt'] is None:
                    continue
                
                branching_prompt = prompt_data['branching_prompt']
                
                # Collect existing data points that have all required fields
                if ('prediction' in branching_prompt and 'KL' in branching_prompt and
                    'info_gain' in branching_prompt and 'weighted_info_gain' in branching_prompt and
                    'h_branching_original' in branching_prompt):
                    
                    all_kl_values.append(branching_prompt['KL'])
                    all_info_gains.append(branching_prompt['info_gain'])
                    all_weighted_info_gains.append(branching_prompt['weighted_info_gain'])
                    all_num_original_conditions.append(branching_prompt['num_original_conditions'])
                    all_h_branching_original.append(branching_prompt['h_branching_original'])
        
        print(f"Found {len(all_kl_values)} existing data points from previous runs")
        
        # Second pass: process new branching prompts (execution)
        print("Processing new branching prompts...")
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
            
            print(f"Processing variable: {var_name} (using {prompts_field})")
            
            for i, prompt_data in enumerate(prompts_data):
                # Skip if no branching prompt
                if 'branching_prompt' not in prompt_data or prompt_data['branching_prompt'] is None:
                    continue
                
                branching_prompt = prompt_data['branching_prompt']
                
                # Skip if already has prediction and KL for branching prompt
                if 'prediction' in branching_prompt and 'KL' in branching_prompt:
                    continue
                
                # Make LLM call for branching prompt
                try:
                    prediction_text = self.single_LLM_call(
                        branching_prompt['prompt'], 
                        self.client, 
                        f"{var_name}_branching_{i}"
                    )
                    
                    # Parse prediction
                    prediction_array = self.parse_prediction(prediction_text)
                    
                    if prediction_array is not None:
                        # Parse completion
                        completion_text = branching_prompt['completion']
                        completion_array = self.parse_prediction(completion_text)
                        
                        if completion_array is not None and len(prediction_array) == len(completion_array):
                            # Compute KL divergence
                            kl_value = self.compute_kl_divergence(completion_array, prediction_array)
                            
                            # Calculate information gain using the SAME logic as GSS_question_quality.py
                            # For branching prompts, we want to use the ORIGINAL target variable, not the branching target
                            # This should match the original prompt's target vs condition structure
                            
                            # Get the original target variable (the main variable being predicted)
                            original_target_var = var_name  # This is the main target variable
                            
                            # Get the original prompt's condition structure
                            original_prompt_conditions = prompt_data['conditions_used']
                            all_condition_vars = list(original_prompt_conditions.keys())
                            all_condition_values = list(original_prompt_conditions.values())
                            
                            # Split exactly like GSS_question_quality.py does:
                            # Original conditions: all except the last
                            original_constraints = {}
                            for j in range(len(all_condition_vars) - 1):
                                original_constraints[all_condition_vars[j]] = all_condition_values[j]
                            
                            # Additional conditions: just the last one
                            if len(all_condition_vars) > 0:
                                last_var = all_condition_vars[-1]
                                last_value = all_condition_values[-1]
                                additional_constraints = {last_var: last_value}
                            else:
                                additional_constraints = {}
                            
                            info_gain, weighted_info_gain = self.calculate_information_metrics(
                                original_target_var, original_constraints, additional_constraints
                            )
                            
                            # Calculate H(new condition | original conditions) for coloring the branching plots
                            # The new condition is the branching target variable
                            branching_target_var = branching_prompt['target_variable']
                            branching_original_conditions = branching_prompt['original_conditions']
                            
                            try:
                                dist_branching_original = self.get_variable_distribution_with_constraints(
                                    branching_target_var, branching_original_conditions
                                )
                                h_branching_original = self.calculate_entropy(dist_branching_original)
                            except (ValueError, KeyError) as e:
                                print(f"  Warning: Could not calculate branching original entropy for {branching_target_var}: {e}")
                                h_branching_original = None
                            
                            # Store results
                            branching_prompt['prediction'] = prediction_text
                            branching_prompt['KL'] = float(kl_value)
                            
                            if info_gain is not None and weighted_info_gain is not None:
                                branching_prompt['info_gain'] = float(info_gain)
                                branching_prompt['weighted_info_gain'] = float(weighted_info_gain)
                                
                                if h_branching_original is not None:
                                    branching_prompt['h_branching_original'] = float(h_branching_original)
                                    
                                    # Add to collection for plotting
                                    all_kl_values.append(kl_value)
                                    all_info_gains.append(info_gain)
                                    all_weighted_info_gains.append(weighted_info_gain)
                                    all_num_original_conditions.append(branching_prompt['num_original_conditions'])
                                    all_h_branching_original.append(h_branching_original)
                                    
                                    print(f"  Branching Prompt {i}: KL = {kl_value:.4f}, Info Gain = {info_gain:.4f}, H(branch|orig) = {h_branching_original:.4f}")
                                else:
                                    print(f"  Branching Prompt {i}: KL = {kl_value:.4f}, Info Gain = {info_gain:.4f}, H(branch|orig) = None")
                            else:
                                print(f"  Branching Prompt {i}: KL = {kl_value:.4f}, Info Gain = None (calculation failed)")
                        else:
                            print(f"  Branching Prompt {i}: Failed to parse arrays or dimension mismatch")
                    else:
                        print(f"  Branching Prompt {i}: Failed to parse prediction")
                        
                except Exception as e:
                    print(f"  Branching Prompt {i}: Error - {str(e)}")
            
            processed_vars += 1
        
        # Save updated data to output file
        with open(output_file_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Total data points available for plotting: {len(all_kl_values)}")
        
        return all_num_original_conditions, all_info_gains, all_weighted_info_gains, all_kl_values, all_h_branching_original, data
    
    def create_condition_specific_plots(self, num_original_conditions, info_gains, weighted_info_gains, kl_values, h_branching_original, data, use_weighted=False, output_dir="./branching_plots"):
        """Create plots for each number of original conditions (0-4)"""
        plot_type = "weighted" if use_weighted else "unweighted"
        output_subdir = os.path.join(output_dir, plot_type)
        os.makedirs(output_subdir, exist_ok=True)
        
        condition_stats = {}
        
        # Determine which metric to use
        x_values = weighted_info_gains if use_weighted else info_gains
        x_label = 'Weighted Information Gain (H(target|original) - H(target|original+more)) * P(more|original)' if use_weighted else 'Information Gain (H(target|original) - H(target|original+more))'
        
        # Create plot for each number of original conditions
        for num_conditions in range(5):  # 0, 1, 2, 3, 4
            # Filter points for this number of conditions
            condition_indices = [i for i, nc in enumerate(num_original_conditions) if nc == num_conditions]
            
            if len(condition_indices) < 2:
                print(f"Skipping {num_conditions} conditions ({plot_type}): insufficient data points ({len(condition_indices)})")
                continue
            
            # Extract data for plotting
            condition_x_values = [x_values[i] for i in condition_indices]
            condition_kl_values = [kl_values[i] for i in condition_indices]
            condition_colors = [h_branching_original[i] for i in condition_indices]  # Use branching entropy for coloring
            
            # Filter out extreme KL values
            filtered_indices = [i for i, kl in enumerate(condition_kl_values) if kl <= 2]
            filtered_x_values = [condition_x_values[i] for i in filtered_indices]
            filtered_kl_values = [condition_kl_values[i] for i in filtered_indices]
            filtered_colors = [condition_colors[i] for i in filtered_indices]
            filtered_count = len(condition_kl_values) - len(filtered_kl_values)
            
            if len(filtered_x_values) < 2:
                print(f"Skipping {num_conditions} conditions ({plot_type}): insufficient data after filtering ({len(filtered_x_values)})")
                continue
            
            # Create the plot
            plt.figure(figsize=(12, 8))
            
            # Create scatter plot with color mapping
            scatter = plt.scatter(filtered_x_values, filtered_kl_values, c=filtered_colors, alpha=0.7, s=80, cmap='viridis')
            
            # Add colorbar
            cbar = plt.colorbar(scatter)
            cbar.set_label('H(new condition | original conditions)', fontsize=12)
            
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
            plt.ylabel('KL Divergence (Branching Prompts)', fontsize=12)
            
            title_prefix = "Weighted Information Gain" if use_weighted else "Information Gain"
            plt.title(f'{title_prefix} vs KL Divergence (Branching): {num_conditions} Original Condition{"s" if num_conditions != 1 else ""}\n(Colored by H(new condition|original), KL ≤ 2)', fontsize=14)
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Add statistics text
            stats_text = f'Points: {len(filtered_x_values)}'
            if filtered_count > 0:
                stats_text += f' (filtered: {filtered_count})'
            if len(filtered_x_values) > 1:
                stats_text += f'\nCorrelation: {correlation:.3f}'
            if filtered_colors:
                stats_text += f'\nAvg H(new|orig): {np.mean(filtered_colors):.3f}'
                
            plt.text(0.02, 0.98, stats_text,
                    transform=plt.gca().transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            # Save the plot
            output_path = os.path.join(output_subdir, f'{plot_type}_branching_info_gain_vs_kl_{num_conditions}_conditions.png')
            plt.tight_layout()
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Created {plot_type} branching plot for {num_conditions} condition{'s' if num_conditions != 1 else ''}: {len(filtered_x_values)} points")
        
        return condition_stats
    
    def create_combined_plot(self, num_original_conditions, info_gains, weighted_info_gains, kl_values, h_branching_original, use_weighted=False, output_dir="./branching_plots"):
        """Create a combined plot with all condition sizes"""
        plot_type = "weighted" if use_weighted else "unweighted"
        output_subdir = os.path.join(output_dir, plot_type)
        
        # Filter out points with KL > 2
        filtered_indices = [i for i, kl in enumerate(kl_values) if kl <= 2]
        if not filtered_indices:
            print(f"No data points remaining after filtering for combined {plot_type} plot")
            return
        
        filtered_out_count = len(kl_values) - len(filtered_indices)
        
        # Determine which metric to use
        x_values = weighted_info_gains if use_weighted else info_gains
        x_label = 'Weighted Information Gain (H(target|original) - H(target|original+more)) * P(more|original)' if use_weighted else 'Information Gain (H(target|original) - H(target|original+more))'
        
        # Extract filtered data for plotting
        filtered_x_values = [x_values[i] for i in filtered_indices]
        filtered_kl_values = [kl_values[i] for i in filtered_indices]
        filtered_num_conditions = [num_original_conditions[i] for i in filtered_indices]
        
        plt.figure(figsize=(12, 10))
        
        # Create scatter plot
        scatter = plt.scatter(filtered_x_values, filtered_kl_values, alpha=0.6, s=60, c='blue')
        
        # Calculate overall correlation and line of best fit
        if len(filtered_x_values) > 1:
            correlation = np.corrcoef(filtered_x_values, filtered_kl_values)[0, 1]
            slope, intercept, r_value, p_value, std_err = stats.linregress(filtered_x_values, filtered_kl_values)
            
            # Plot line of best fit
            x_line = np.linspace(min(filtered_x_values), max(filtered_x_values), 100)
            y_line = slope * x_line + intercept
            plt.plot(x_line, y_line, 'r-', alpha=0.8, linewidth=2,
                    label=f'Best fit (R² = {r_value**2:.3f})')
        
        plt.xlabel(x_label, fontsize=12)
        plt.ylabel('KL Divergence (Branching Prompts)', fontsize=12)
        
        title_prefix = "Weighted Information Gain" if use_weighted else "Information Gain"
        plt.title(f'{title_prefix} vs KL Divergence (Branching): All Original Condition Sizes Combined\n(KL ≤ 2)', fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Add statistics text
        stats_text = f'Points: {len(filtered_x_values)}'
        if filtered_out_count > 0:
            stats_text += f' (filtered: {filtered_out_count})'
        if len(filtered_x_values) > 1:
            stats_text += f'\nCorrelation: {correlation:.3f}'
        
        # Count points by condition size
        condition_counts = {}
        for nc in filtered_num_conditions:
            condition_counts[nc] = condition_counts.get(nc, 0) + 1
        
        condition_breakdown = ', '.join([f'{k}: {v}' for k, v in sorted(condition_counts.items())])
        stats_text += f'\nBy condition: {condition_breakdown}'
            
        plt.text(0.02, 0.98, stats_text,
                transform=plt.gca().transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Save the plot
        output_path = os.path.join(output_subdir, f'{plot_type}_branching_info_gain_vs_kl_combined.png')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Created combined {plot_type} branching plot with {len(filtered_x_values)} total points")
        
        return {
            'correlation': correlation if len(filtered_x_values) > 1 else None,
            'r_squared': r_value**2 if len(filtered_x_values) > 1 else None,
            'slope': slope if len(filtered_x_values) > 1 else None,
            'intercept': intercept if len(filtered_x_values) > 1 else None,
            'n_points': len(filtered_x_values),
            'filtered_points': filtered_out_count
        }

def main():
    # Configuration
    API_KEY = 'YOUR_OPENAI_API_KEY'  # Replace with your actual API key
    INPUT_FILE = "/Users/ruoxiliu/Desktop/Argo_research/reasoning/sense/backtracking/gss2024_mapping_relevant_branch_executed.json"  # Replace with your input file path
    OUTPUT_FILE = "/Users/ruoxiliu/Desktop/Argo_research/reasoning/sense/backtracking/gss2024_mapping_relevant_branch_executed_1.json"  # Output file for processed data
    CSV_FILE = "gss_2022.csv"  # GSS dataset file
    OUTPUT_DIR = "./branching_performance_1"  # Directory for all plots
    MAX_VAR = 100  # Set to None to process all variables

    
    # Initialize processor
    processor = BranchingPromptProcessor(API_KEY, CSV_FILE)
    
    # Process the branching prompts
    print("Processing branching prompts...")
    # FIX: The function returns 6 values, not 5
    num_original_conditions, info_gains, weighted_info_gains, kl_values, h_branching_original, processed_data = processor.process_branching_prompts(
        INPUT_FILE, 
        OUTPUT_FILE,
        max_var=MAX_VAR
    )
    
    print(f"Saved processed data to: {OUTPUT_FILE}")
    
    if len(kl_values) > 0:
        # Create output directory
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        # Create both sets of plots
        print(f"\n📊 Creating UNWEIGHTED plots (H(target|original) - H(target|original+more))...")
        unweighted_condition_stats = processor.create_condition_specific_plots(
            num_original_conditions, info_gains, weighted_info_gains, kl_values, h_branching_original, 
            use_weighted=False, output_dir=OUTPUT_DIR
        )
        unweighted_combined_stats = processor.create_combined_plot(
            num_original_conditions, info_gains, weighted_info_gains, kl_values, h_branching_original,
            use_weighted=False, output_dir=OUTPUT_DIR
        )
        
        print(f"\n📊 Creating WEIGHTED plots ((H(target|original) - H(target|original+more)) * P(more|original))...")
        weighted_condition_stats = processor.create_condition_specific_plots(
            num_original_conditions, info_gains, weighted_info_gains, kl_values, h_branching_original,
            use_weighted=True, output_dir=OUTPUT_DIR
        )
        weighted_combined_stats = processor.create_combined_plot(
            num_original_conditions, info_gains, weighted_info_gains, kl_values, h_branching_original,
            use_weighted=True, output_dir=OUTPUT_DIR
        )
        
        # Print summary statistics
        print(f"\n📊 BRANCHING PROMPT ANALYSIS RESULTS:")
        print(f"Total data points: {len(kl_values)}")
        
        print(f"\n🔸 UNWEIGHTED Information Gain Results (Branching Prompts):")
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
        
        print(f"\n🔹 WEIGHTED Information Gain Results (Branching Prompts):")
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
        
        # Basic statistics
        print(f"\nBranching Prompt KL Divergence Statistics:")
        print(f"  Mean: {np.mean(kl_values):.4f}")
        print(f"  Std: {np.std(kl_values):.4f}")
        print(f"  Min: {np.min(kl_values):.4f}")
        print(f"  Max: {np.max(kl_values):.4f}")
        
        # Statistics for filtered data
        filtered_kl_values = [kl for kl in kl_values if kl <= 2.0]
        if filtered_kl_values:
            print(f"\nFiltered Branching KL Divergence Statistics (KL ≤ 2):")
            print(f"  Count: {len(filtered_kl_values)} / {len(kl_values)}")
            print(f"  Mean: {np.mean(filtered_kl_values):.4f}")
            print(f"  Std: {np.std(filtered_kl_values):.4f}")
            print(f"  Min: {np.min(filtered_kl_values):.4f}")
            print(f"  Max: {np.max(filtered_kl_values):.4f}")
        
        print(f"\nInformation Gain Statistics:")
        print(f"  Unweighted - Mean: {np.mean(info_gains):.4f}, Std: {np.std(info_gains):.4f}")
        print(f"  Weighted - Mean: {np.mean(weighted_info_gains):.4f}, Std: {np.std(weighted_info_gains):.4f}")
        
        print(f"\nAll plots saved to directory: {OUTPUT_DIR}")
        print(f"  📁 Unweighted plots: {os.path.join(OUTPUT_DIR, 'unweighted')}")
        print(f"  📁 Weighted plots: {os.path.join(OUTPUT_DIR, 'weighted')}")
        
        # List created files
        for plot_type in ['unweighted', 'weighted']:
            subdir = os.path.join(OUTPUT_DIR, plot_type)
            if os.path.exists(subdir):
                plot_files = [f for f in os.listdir(subdir) if f.endswith('.png')]
                print(f"\n{plot_type.title()} plots ({len(plot_files)} files):")
                for plot_file in sorted(plot_files):
                    print(f"  - {plot_file}")
                    
    else:
        print("No branching prompt data points found to plot.")

if __name__ == "__main__":
    main()