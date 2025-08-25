import json
import sys
import os
import random
from pathlib import Path
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional

def load_gss_dataset(file_path):
    """Load the GSS dataset from various possible formats"""
    print(f"Loading GSS dataset from: {file_path}")
    
    file_ext = Path(file_path).suffix.lower()
    
    try:
        if file_ext == '.csv':
            for encoding in ['utf-8', 'latin-1', 'cp1252']:
                try:
                    df = pd.read_csv(file_path, encoding=encoding, low_memory=False)
                    print(f"✓ Successfully loaded CSV with {encoding} encoding")
                    break
                except UnicodeDecodeError:
                    continue
            else:
                raise ValueError("Could not read CSV with any common encoding")
                
        elif file_ext in ['.dta', '.stata']:
            df = pd.read_stata(file_path)
            print("✓ Successfully loaded Stata file")
            
        elif file_ext in ['.sav', '.spss']:
            try:
                import pyreadstat
                df, meta = pyreadstat.read_sav(file_path)
                print("✓ Successfully loaded SPSS file")
            except ImportError:
                print("❌ pyreadstat required for SPSS files. Install with: pip install pyreadstat")
                sys.exit(1)
                
        elif file_ext in ['.xlsx', '.xls']:
            df = pd.read_excel(file_path)
            print("✓ Successfully loaded Excel file")
            
        else:
            df = pd.read_csv(file_path, low_memory=False)
            print("✓ Successfully loaded as CSV (default)")
        
        df.columns = df.columns.str.lower()
        print(f"Dataset shape: {df.shape}")
        return df
        
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        sys.exit(1)

def load_mappings(mapping_file):
    """Load the variable mappings"""
    print(f"Loading mappings from: {mapping_file}")
    
    try:
        with open(mapping_file, 'r', encoding='utf-8') as f:
            mappings = json.load(f)
        print(f"✓ Loaded {len(mappings)} variables from mapping file")
        return mappings
    except Exception as e:
        print(f"❌ Error loading mappings: {e}")
        sys.exit(1)

def get_variables_with_related_vars(mappings, min_related_vars=1):
    """Get variables that have related_variables field with enough variables"""
    valid_vars = []
    
    for var_name, var_info in mappings.items():
        # Check if has related_variables field
        if 'related_variables' not in var_info:
            continue
            
        related_vars = var_info['related_variables']
        if not isinstance(related_vars, list) or len(related_vars) < min_related_vars:
            continue
            
        # Check if has valid question and domain
        if 'revised_question' not in var_info or not var_info['revised_question']:
            continue
            
        if 'domain_values' not in var_info or not var_info['domain_values']:
            continue
            
        # Filter out invalid domain values
        valid_domain = get_valid_domain_values(var_info['domain_values'])
        if len(valid_domain) < 2:  # Need at least 2 valid options
            continue
            
        # Filter out domains with more than 7 values
        if len(valid_domain) > 7:
            continue
            
        valid_vars.append(var_name)
    
    print(f"Found {len(valid_vars)} variables with related_variables and valid domains")
    return valid_vars

def get_valid_domain_values(domain_values):
    """Filter out invalid response types from domain values"""
    invalid_responses = {
        'not available in this year', 'not available this year', 'not avail in this year',
        'not avail this year', 'nay', 'na', 'n/a', 'iap', 'dk', "don't know",
        'no answer', 'skipped on web', 'refused', '.', 'missing', 'not applicable'
    }
    
    valid_domain = []
    for val in domain_values:
        if str(val).lower().strip() not in invalid_responses:
            valid_domain.append(val)
    
    return valid_domain

def count_samples_after_conditions(df, condition_vars, condition_values):
    """Count how many samples remain after applying conditions"""
    # Use most recent year available
    if 'year' in df.columns:
        max_year = df['year'].max()
        df_filtered = df[df['year'] == max_year]
    else:
        df_filtered = df.copy()
    
    if len(df_filtered) == 0:
        return 0
    
    # Apply conditions
    for var, value in zip(condition_vars, condition_values):
        if var in df_filtered.columns:
            df_filtered = df_filtered[df_filtered[var] == value]
    
    return len(df_filtered)

def get_answer_distribution(df, target_var, condition_vars, condition_values, target_domain_values):
    """
    Get the answer distribution for target_var given conditions from the GSS dataset
    """
    # Use most recent year available
    if 'year' in df.columns:
        max_year = df['year'].max()
        df_filtered = df[df['year'] == max_year]
        print(f"  Using data from year {max_year}")
    else:
        df_filtered = df.copy()
    
    if len(df_filtered) == 0:
        return None
    
    # Apply conditions
    for var, value in zip(condition_vars, condition_values):
        if var in df_filtered.columns:
            df_filtered = df_filtered[df_filtered[var] == value]
    
    if len(df_filtered) == 0:
        print(f"  No data after applying conditions")
        return None
    
    # Get target variable responses
    if target_var not in df_filtered.columns:
        print(f"  Target variable {target_var} not found in dataset")
        return None
    
    target_responses = df_filtered[target_var].dropna()
    
    # Invalid responses to exclude
    invalid_responses = {
        'not available in this year', 'not available this year', 'not avail in this year',
        'not avail this year', 'nay', 'na', 'n/a', 'iap', 'dk', "don't know",
        'no answer', 'skipped on web', 'refused', '.', 'missing', 'not applicable'
    }
    
    # Filter out invalid responses
    valid_responses = []
    for response in target_responses:
        if str(response).lower().strip() not in invalid_responses:
            valid_responses.append(response)
    
    if len(valid_responses) == 0:
        print(f"  No valid responses after filtering")
        return None
    
    # Count occurrences of each domain value
    domain_counts = {}
    for domain_val in target_domain_values:
        domain_counts[domain_val] = 0
    
    # Count valid responses
    for response in valid_responses:
        if response in target_domain_values:
            domain_counts[response] += 1
    
    total_valid = sum(domain_counts.values())
    if total_valid == 0:
        print(f"  No responses match domain values")
        return None
    
    # Convert to probabilities
    probabilities = []
    for domain_val in target_domain_values:
        prob = domain_counts[domain_val] / total_valid
        probabilities.append(round(prob, 4))
    
    print(f"  Distribution based on {total_valid} valid responses: {probabilities}")
    return probabilities

def select_k_value(k_mode):
    """Select k value based on the mode specified"""
    if k_mode == "fixed":
        return 1  # Fixed value
    elif k_mode == "random":
        return random.choice([1, 2, 3])  # Random from set
    else:
        # Assume it's a specific number
        try:
            return int(k_mode)
        except:
            return 1

def generate_input_line(df, target_var, target_info, related_vars, mappings, k_value, min_samples=50):
    """Generate a single input line for the maieutic prompting workflow"""
    
    # Get target question and domain
    target_question = target_info['revised_question']
    target_domain = get_valid_domain_values(target_info['domain_values'])
    
    if len(target_domain) < 2:
        return None
    
    # Format domain for B part
    domain_str = "; ".join(target_domain)
    b_part = f"{target_question} [{domain_str}]"
    
    # Select k related variables
    available_related = []
    for rel_var in related_vars:
        if rel_var in mappings and 'revised_question' in mappings[rel_var]:
            available_related.append(rel_var)
    
    if len(available_related) == 0:
        return None
    
    # Try different values of k, starting from k_value and going down
    # Stop when we have enough samples or reach k=1
    for current_k in range(k_value, 0, -1):
        k_to_use = min(current_k, len(available_related))
        selected_related = available_related[:k_to_use]
        
        # Get conditions for A part
        condition_vars = []
        condition_values = []
        condition_questions = []
        
        for rel_var in selected_related:
            rel_info = mappings[rel_var]
            if 'revised_question' not in rel_info or 'domain_values' not in rel_info:
                continue
                
            rel_domain = get_valid_domain_values(rel_info['domain_values'])
            if len(rel_domain) == 0:
                continue
            
            # Use first domain value as the condition (or random selection)
            condition_value = rel_domain[0]  # Could be randomized
            
            condition_vars.append(rel_var)
            condition_values.append(condition_value)
            condition_questions.append(rel_info['revised_question'])
        
        if len(condition_vars) == 0:
            continue
        
        # Check sample size before proceeding
        sample_count = count_samples_after_conditions(df, condition_vars, condition_values)
        
        if sample_count >= min_samples or current_k == 1:  # Use this k or if it's the last try
            print(f"  Using k={len(condition_vars)} (sample count: {sample_count})")
            
            # Create A part - format as list if multiple conditions
            if len(condition_vars) == 1:
                a_part = f"{condition_questions[0]} {condition_values[0]}"
            else:
                a_conditions = []
                for question, value in zip(condition_questions, condition_values):
                    a_conditions.append(f'"{question} {value}"')
                a_part = "[" + ", ".join(a_conditions) + "]"
            
            # Get the true distribution from GSS data
            distribution = get_answer_distribution(df, target_var, condition_vars, condition_values, target_domain)
            
            if distribution is None:
                continue
            
            # Check that domain and distribution have same length
            if len(target_domain) != len(distribution):
                print(f"  ERROR: Domain length {len(target_domain)} != distribution length {len(distribution)}")
                continue
            
            # Format distribution
            dist_str = "; ".join([f"{prob:.4f}" for prob in distribution])
            
            # Create the input line
            input_line = f"{{B: {b_part}; A: {a_part}; [{dist_str}]}}"
            
            return input_line
        else:
            print(f"  k={current_k} has only {sample_count} samples (< {min_samples}), trying k={current_k-1}")
    
    return None

def generate_maieutic_input_file(gss_file, mapping_file, output_file, k_mode="fixed", max_variables=None):
    """Generate input file for maieutic prompting workflow"""
    
    print("Maieutic Input File Generator")
    print("=" * 50)
    print(f"GSS dataset: {gss_file}")
    print(f"Mapping file: {mapping_file}")
    print(f"Output file: {output_file}")
    print(f"K mode: {k_mode}")
    if max_variables:
        print(f"Max variables: {max_variables}")
    print()
    
    # Load data
    df = load_gss_dataset(gss_file)
    mappings = load_mappings(mapping_file)
    
    # Get valid variables
    valid_vars = get_variables_with_related_vars(mappings, min_related_vars=1)
    
    if len(valid_vars) == 0:
        print("❌ No valid variables found with related_variables")
        return
    
    # Limit variables if specified
    if max_variables and max_variables < len(valid_vars):
        valid_vars = valid_vars[:max_variables]
        print(f"Processing first {len(valid_vars)} variables")
    
    print(f"Processing {len(valid_vars)} variables...")
    print("-" * 50)
    
    # Generate input lines
    input_lines = []
    successful = 0
    failed = 0
    
    for i, target_var in enumerate(valid_vars):
        print(f"[{i+1:3d}/{len(valid_vars)}] Processing {target_var}...")
        
        try:
            target_info = mappings[target_var]
            related_vars = target_info['related_variables']
            
            # Select k value
            k_value = select_k_value(k_mode)
            
            # Generate input line
            input_line = generate_input_line(df, target_var, target_info, related_vars, mappings, k_value)
            
            if input_line:
                input_lines.append(input_line)
                successful += 1
                print(f"  ✓ Generated input line (k={k_value})")
                
                # Show preview
                if len(input_line) > 100:
                    preview = input_line[:100] + "..."
                else:
                    preview = input_line
                print(f"  Preview: {preview}")
            else:
                failed += 1
                print(f"  ✗ Failed to generate input line")
                
        except Exception as e:
            failed += 1
            print(f"  ❌ Error processing {target_var}: {e}")
    
    # Write output file
    if input_lines:
        print(f"\nWriting {len(input_lines)} input lines to {output_file}")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for line in input_lines:
                f.write(line + '\n')
        
        print(f"✓ Successfully wrote {len(input_lines)} lines to {output_file}")
    else:
        print("❌ No input lines generated")
    
    # Print statistics
    print(f"\n📊 GENERATION COMPLETE")
    print(f"Total processed: {successful + failed}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    if successful + failed > 0:
        success_rate = successful / (successful + failed) * 100
        print(f"Success rate: {success_rate:.1f}%")

def test_single_variable(gss_file, mapping_file, target_var, k_mode="fixed"):
    """Test input generation for a single variable"""
    
    print(f"Testing input generation for: {target_var}")
    print("=" * 50)
    
    # Load data
    df = load_gss_dataset(gss_file)
    mappings = load_mappings(mapping_file)
    
    if target_var not in mappings:
        print(f"❌ Variable {target_var} not found in mappings")
        return
    
    target_info = mappings[target_var]
    
    # Check if has related_variables
    if 'related_variables' not in target_info:
        print(f"❌ Variable {target_var} has no related_variables field")
        return
    
    related_vars = target_info['related_variables']
    if not isinstance(related_vars, list) or len(related_vars) == 0:
        print(f"❌ Variable {target_var} has no valid related_variables")
        return
    
    print(f"Target variable: {target_var}")
    print(f"Question: {target_info.get('revised_question', 'N/A')}")
    print(f"Domain: {target_info.get('domain_values', [])}")
    print(f"Related variables: {related_vars}")
    print()
    
    # Generate input line
    k_value = select_k_value(k_mode)
    input_line = generate_input_line(df, target_var, target_info, related_vars, mappings, k_value)
    
    if input_line:
        print(f"✓ Generated input line (k={k_value}):")
        print(input_line)
    else:
        print(f"✗ Failed to generate input line")

def main():
    """Main function with hardcoded file paths and settings"""
    
    # ===== CONFIGURATION - MODIFY THESE PATHS AND SETTINGS =====
    
    # File paths
    gss_file = "/Users/ruoxiliu/Desktop/Argo_research/reasoning/sense/backtracking/gss_2022.csv"
    mapping_file = "/Users/ruoxiliu/Desktop/Argo_research/reasoning/sense/backtracking/gss2024_mapping_relevant_executed.json"
    output_file = "/Users/ruoxiliu/Desktop/Argo_research/reasoning/sense/backtracking/experiment_maieutic_nonbinary/input_2.txt"
    
    # Settings
    k_mode = "3"  # Options: "fixed", "random", or specific number like "2"
    max_variables = None  # Set to None for no limit
    
    # Test mode settings (set test_mode = True to test single variable)
    test_mode = False
    test_target_var = "ptnrornt"  # Variable to test if test_mode is True
    
    # ============================================================
    
    print("Maieutic Input File Generator")
    print("=" * 50)
    print(f"GSS dataset: {gss_file}")
    print(f"Mapping file: {mapping_file}")
    print(f"Output file: {output_file}")
    print(f"K mode: {k_mode}")
    print(f"Max variables: {max_variables}")
    print(f"Test mode: {test_mode}")
    if test_mode:
        print(f"Test variable: {test_target_var}")
    print()
    
    # Validate input files
    for file_path in [gss_file, mapping_file]:
        if not os.path.exists(file_path):
            print(f"❌ File not found: {file_path}")
            sys.exit(1)
    
    # Run the appropriate mode
    try:
        if test_mode:
            test_single_variable(gss_file, mapping_file, test_target_var, k_mode)
        else:
            generate_maieutic_input_file(gss_file, mapping_file, output_file, k_mode, max_variables)
        
    except KeyboardInterrupt:
        print("\n⚠️ Process interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error during processing: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()