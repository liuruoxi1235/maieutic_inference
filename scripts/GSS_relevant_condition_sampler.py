import json
import sys
import os
import itertools
from pathlib import Path
import pandas as pd
import numpy as np
from openai import OpenAI
import time

class GSSSmartConditionsGenerator:
    def __init__(self, api_key, model_id="gpt-4o", temperature=0.3, log_path="gss_smart_conditions.log"):
        """
        Initialize the GSS Smart Conditions Generator
        """
        self.client = OpenAI(api_key=api_key)
        self.model_ID = model_id
        self.temperature = temperature
        self.log_pth = log_path
        
        # Clear log file
        with open(self.log_pth, "w") as log_file:
            log_file.write("GSS Smart Conditions Generation Log\n" + "="*50 + "\n\n")
    
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
        
        with open(self.log_pth, "a") as log_file:
            log_file.write(f"[main {name}] [response] : {response.choices[0].message.content}\n\n")

        # Return the actual generation
        return response.choices[0].message.content

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
    """Load the variable mappings with revised questions"""
    print(f"Loading mappings from: {mapping_file}")
    
    try:
        with open(mapping_file, 'r', encoding='utf-8') as f:
            mappings = json.load(f)
        print(f"✓ Loaded {len(mappings)} variables from mapping file")
        return mappings
    except Exception as e:
        print(f"❌ Error loading mappings: {e}")
        sys.exit(1)

def update_incomplete_domains(df, mappings):
    """
    Update domain values for discrete variables by replacing them with actual values from 2022 GSS dataset
    Only processes variables with data_type = 'discrete'
    """
    print("Updating domains for discrete variables from 2022 GSS data...")
    
    # Filter for 2022 data
    df_2022 = df[df['year'] == 2022] if 'year' in df.columns else df
    
    if len(df_2022) == 0:
        print("❌ No 2022 data found")
        return mappings
    
    updated_count = 0
    
    # Get discrete variables that are accepted
    excluded_patterns = ['ethregion', 'ethworld', 'raceacs']
    discrete_variables = []
    for var_name, var_info in mappings.items():
        if (var_info.get('revision_status') == 'accepted' and 
            var_info.get('data_type') == 'discrete'):
            if not any(pattern in var_name.lower() for pattern in excluded_patterns):
                discrete_variables.append(var_name)
    
    print(f"Checking domains for {len(discrete_variables)} discrete variables...")
    
    for var_name in discrete_variables:
        if var_name not in df_2022.columns:
            continue
            
        var_info = mappings[var_name]
        current_domain = var_info.get('domain_values', [])
        
        # Get all unique values from 2022 GSS data for this variable
        actual_values = df_2022[var_name].dropna().unique()
        actual_values = sorted([str(val) for val in actual_values])  # Convert to strings and sort
        
        # Convert current domain to strings for comparison
        current_domain_str = sorted([str(val) for val in current_domain])
        
        # Check if domains are different
        if set(actual_values) != set(current_domain_str):
            # Replace the entire domain with actual GSS values
            mappings[var_name]['domain_values'] = actual_values
            updated_count += 1
            
            added_values = set(actual_values) - set(current_domain_str)
            removed_values = set(current_domain_str) - set(actual_values)
            
            print(f"  Updated {var_name}:")
            print(f"    New domain: {len(actual_values)} values")
            if added_values:
                print(f"    Added: {sorted(list(added_values))}")
            if removed_values:
                print(f"    Removed: {sorted(list(removed_values))}")
    
    if updated_count > 0:
        print(f"✓ Updated domains for {updated_count} discrete variables")
    else:
        print("✓ All discrete variable domains are accurate - no updates needed")
    
    return mappings

def get_eligible_variables(mappings):
    """
    Get variables that are eligible for processing:
    1. Have revision_status = 'accepted'
    2. Don't contain excluded patterns in variable name
    3. Either don't have relevant_constraints OR have relevant_constraints = null (for retry)
    """
    excluded_patterns = ['ethregion', 'ethworld', 'raceacs']
    eligible_vars = []
    already_processed = []
    retry_vars = []
    
    for var_name, var_info in mappings.items():
        if var_info.get('revision_status') == 'accepted':
            if not any(pattern in var_name.lower() for pattern in excluded_patterns):
                # Check processing status
                if 'relevant_constraints' not in var_info:
                    # Never processed
                    eligible_vars.append(var_name)
                elif var_info['relevant_constraints'] is None:
                    # Previously failed, eligible for retry
                    eligible_vars.append(var_name)
                    retry_vars.append(var_name)
                else:
                    # Successfully processed
                    already_processed.append(var_name)
    
    print(f"Found {len(eligible_vars)} eligible variables for processing (after filtering)")
    if already_processed:
        print(f"Skipped {len(already_processed)} variables that already have relevant_constraints")
    if retry_vars:
        print(f"Including {len(retry_vars)} variables for retry (previously failed)")
    
    return eligible_vars

def get_all_condition_candidates(mappings):
    """
    Get all variables that can be used as conditions (including already processed ones):
    1. Have revision_status = 'accepted'  
    2. Don't contain excluded patterns in variable name
    Note: This includes variables that already have relevant_constraints
    """
    excluded_patterns = ['ethregion', 'ethworld', 'raceacs']
    condition_candidates = []
    
    for var_name, var_info in mappings.items():
        if var_info.get('revision_status') == 'accepted':
            if not any(pattern in var_name.lower() for pattern in excluded_patterns):
                condition_candidates.append(var_name)
    
    return condition_candidates

def create_variable_bins(variables, bin_size=50):
    """Split variables into bins of specified size"""
    bins = []
    for i in range(0, len(variables), bin_size):
        bin_vars = variables[i:i + bin_size]
        bins.append(bin_vars)
    
    print(f"Created {len(bins)} bins with up to {bin_size} variables each")
    return bins

def get_variable_selection_prompt(target_var, target_question, bin_variables, mappings):
    """Create prompt for LLM to select related variables from a bin"""
    
    # Create variable descriptions for the bin
    var_descriptions = []
    for var in bin_variables:
        if var in mappings:
            question = mappings[var].get('revised_question', 'No description')
            data_type = mappings[var].get('data_type', 'unknown')
            domain_values = mappings[var].get('domain_values', [])[:3]  # Show first 3 domain values
            domain_str = ', '.join(domain_values) + ('...' if len(mappings[var].get('domain_values', [])) > 3 else '')
            
            var_descriptions.append(f"- {var}: {question} (Type: {data_type}, Examples: {domain_str})")
    
    variables_text = '\n'.join(var_descriptions)
    
    return [
        ("system", """You are an expert survey methodologist and social scientist. Your task is to identify variables that are most related to a target variable from a survey dataset.

Given a target variable and a list of candidate variables, select the 5 most helpful conditional variables. Consider variables that:

1. **Value of Information**: If this conditional variable is known, then you'll be more certain about what the target variable's answer is likely going to be
3. **Theoretical connections**: Variables that social science theory suggests should be related
4. **Specification of situation**: Variables that, once known, pinpoints a specific subset of the respondent population, who would have more predictable answers to the target variable
         
IMPORTANT OUTPUT FORMAT:
You must respond with exactly 5 variable names, one per line, in descending order of helpfulness (most helpful and relevant first). 
Use ONLY the variable names provided in the candidate list.
Do not include any explanations, numbers, or additional text.

Example output:
education
income
age  
race
marital"""),
        
        ("user", """TARGET VARIABLE: {target_var}
TARGET QUESTION: {target_question}

CANDIDATE VARIABLES:
{variables_text}

Please select the 5 most conceptually related and helpful variables from the candidate list above, in descending order of helpfulness.""")
    ]

def select_related_variables_from_bin(generator, target_var, target_question, bin_variables, mappings, bin_number):
    """Use LLM to select 5 most related variables from a bin"""
    
    if len(bin_variables) == 0:
        return []
    
    # Remove target variable if it's in the bin
    bin_variables = [v for v in bin_variables if v != target_var]
    
    if len(bin_variables) == 0:
        return []
    
    # Create prompt
    prompt_template = get_variable_selection_prompt(target_var, target_question, bin_variables, mappings)
    
    # Prepare replacements
    var_descriptions = []
    for var in bin_variables:
        if var in mappings:
            question = mappings[var].get('revised_question', 'No description')
            data_type = mappings[var].get('data_type', 'unknown')
            domain_values = mappings[var].get('domain_values', [])[:3]
            domain_str = ', '.join(domain_values) + ('...' if len(mappings[var].get('domain_values', [])) > 3 else '')
            var_descriptions.append(f"- {var}: {question} (Type: {data_type}, Examples: {domain_str})")
    
    replacements = {
        "target_var": target_var,
        "target_question": target_question,
        "variables_text": '\n'.join(var_descriptions)
    }
    
    try:
        print(f"    Querying LLM for bin {bin_number} ({len(bin_variables)} variables)...")
        
        response = generator.single_LLM_call(
            prompt_template,
            generator.client,
            f"select_vars_bin_{bin_number}",
            replacements
        )
        
        # Parse response - expect 5 variable names, one per line
        selected_vars = []
        lines = response.strip().split('\n')
        
        for line in lines:
            var_name = line.strip()
            # Validate that the variable is in our bin
            if var_name in bin_variables:
                selected_vars.append(var_name)
            
        print(f"    ✓ Selected {len(selected_vars)} variables from bin {bin_number}: {selected_vars}")
        
        # Ensure we return exactly 5 (or fewer if bin is small)
        return selected_vars[:5]
        
    except Exception as e:
        print(f"    ❌ Error selecting from bin {bin_number}: {e}")
        return []

def get_30_related_variables(generator, target_var, eligible_vars, mappings):
    """
    Get 30 related variables by:
    1. Splitting eligible vars into bins of 50
    2. Using LLM to select top 5 from each bin  
    3. Organizing by rank (all rank 1s first, then rank 2s, etc.)
    """
    
    # Remove target variable from eligible vars
    candidate_vars = [v for v in eligible_vars if v != target_var]
    
    if len(candidate_vars) < 30:
        print(f"Warning: Only {len(candidate_vars)} candidate variables available")
    
    # Create bins of 50 variables each
    bins = create_variable_bins(candidate_vars, bin_size=50)
    
    target_question = mappings[target_var].get('revised_question', 'No description')
    print(f"Target: {target_var} - {target_question}")
    
    # Get selections from each bin
    all_selections = []
    for i, bin_vars in enumerate(bins):
        selected = select_related_variables_from_bin(
            generator, target_var, target_question, bin_vars, mappings, i+1
        )
        all_selections.append(selected)
        time.sleep(0.5)  # Rate limiting
    
    # Organize by rank: all rank 1s first, then rank 2s, etc.
    organized_vars = []
    max_rank = max(len(selection) for selection in all_selections) if all_selections else 0
    
    for rank in range(max_rank):
        for bin_selection in all_selections:
            if rank < len(bin_selection):
                organized_vars.append(bin_selection[rank])
    
    print(f"✓ Organized {len(organized_vars)} related variables by rank")
    return organized_vars[:30]  # Ensure max 30

def test_5_variable_combination(df, target_var, condition_vars, min_samples=30):
    """Test if a 5-variable combination yields enough samples in 2022 data"""
    
    # Filter for 2022 data
    df_2022 = df[df['year'] == 2022] if 'year' in df.columns else df
    
    if len(df_2022) == 0:
        return False, 0, {}
    
    # Invalid responses to exclude
    invalid_responses = {
        'not available in this year', 'not available this year', 'not avail in this year',
        'not avail this year', 'nay', 'na', 'n/a', 'iap', 'dk', "don't know",
        'no answer', 'skipped on web', 'refused', '.'
    }
    
    current_df = df_2022.copy()
    conditions = {}
    
    # For each condition variable, find the value that maximizes remaining samples
    for cond_var in condition_vars:
        if cond_var not in current_df.columns:
            return False, 0, {}
        
        # Get valid values for this variable
        valid_values = []
        for val in current_df[cond_var].dropna().unique():
            if str(val).lower().strip() not in invalid_responses:
                valid_values.append(val)
        
        if not valid_values:
            return False, 0, {}
        
        # Find value that maximizes remaining samples
        best_value = None
        best_count = 0
        
        for value in valid_values:
            test_df = current_df[current_df[cond_var] == value]
            count = len(test_df)
            if count > best_count:
                best_count = count
                best_value = value
        
        if best_value is None:
            return False, 0, {}
        
        # Apply this condition
        current_df = current_df[current_df[cond_var] == best_value]
        conditions[cond_var] = best_value
        
        if len(current_df) == 0:
            return False, 0, conditions
    
    # Count valid responses for target variable
    if target_var not in current_df.columns:
        return False, 0, conditions
    
    target_values = current_df[target_var].dropna()
    valid_count = 0
    
    for val in target_values:
        if str(val).lower().strip() not in invalid_responses:
            valid_count += 1
    
    return valid_count >= min_samples, valid_count, conditions

def find_best_5_variable_combination(df, target_var, related_vars, min_samples=30, max_combinations=15):
    """
    Find the best 5-variable combination from related vars.
    Tries combinations in order of index sum (lowest first).
    If no 5-variable combination meets min_samples, falls back to best 4-variable combination.
    
    Args:
        max_combinations: Maximum number of 5-variable combinations to test (default: 15)
    """
    
    print(f"Finding best 5-variable combination from {len(related_vars)} related variables...")
    
    # Generate all 5-variable combinations and sort by index sum
    combinations_with_sum = []
    for combo in itertools.combinations(enumerate(related_vars), 5):
        indices = [i for i, var in combo]
        variables = [var for i, var in combo]
        index_sum = sum(indices)
        combinations_with_sum.append((index_sum, indices, variables))
    
    # Sort by index sum (lowest first)
    combinations_with_sum.sort(key=lambda x: x[0])
    
    # Limit the number of combinations to test
    total_combinations = len(combinations_with_sum)
    combinations_to_test = min(max_combinations, total_combinations)
    
    print(f"Testing first {combinations_to_test} of {total_combinations} 5-variable combinations...")
    
    # Track the best 4-variable combination as fallback
    best_4_var_combination = None
    best_4_var_sample_count = 0
    best_4_var_conditions = {}
    
    for attempt in range(combinations_to_test):
        index_sum, indices, variables = combinations_with_sum[attempt]
        
        if attempt % 10 == 0 and attempt > 0:
            print(f"  Tested {attempt} combinations...")
        
        success, sample_count, conditions = test_5_variable_combination(
            df, target_var, variables, min_samples
        )
        
        if success:
            print(f"✓ Found valid 5-variable combination: {variables}")
            print(f"  Indices: {indices}, Sum: {index_sum}")
            print(f"  Sample count: {sample_count}")
            print(f"  Conditions: {conditions}")
            return variables, conditions, sample_count
        
        # During 5-variable testing, also track best 4-variable subsets
        # Test all possible 4-variable subsets of this 5-variable combination
        for four_var_combo in itertools.combinations(variables, 4):
            four_success, four_count, four_conditions = test_5_variable_combination(
                df, target_var, list(four_var_combo), 0  # No minimum for tracking best
            )
            
            if four_count > best_4_var_sample_count:
                best_4_var_combination = list(four_var_combo)
                best_4_var_sample_count = four_count
                best_4_var_conditions = four_conditions
    
    print(f"✗ No 5-variable combination found with {min_samples}+ samples in first {combinations_to_test} attempts")
    
    # Fallback to best 4-variable combination
    if best_4_var_combination and best_4_var_sample_count >= min_samples:
        print(f"💡 Using best 4-variable fallback: {best_4_var_combination}")
        print(f"  Sample count: {best_4_var_sample_count}")
        print(f"  Conditions: {best_4_var_conditions}")
        return best_4_var_combination, best_4_var_conditions, best_4_var_sample_count
    elif best_4_var_combination:
        print(f"💡 Using best 4-variable fallback even though it has only {best_4_var_sample_count} samples")
        print(f"  Combination: {best_4_var_combination}")
        print(f"  Conditions: {best_4_var_conditions}")
        return best_4_var_combination, best_4_var_conditions, best_4_var_sample_count
    
    print(f"✗ No valid combination found in first {combinations_to_test} attempts (best 4-variable had {best_4_var_sample_count} samples)")
    return None, None, 0

def process_single_variable(generator, df, target_var, eligible_vars, mappings, min_samples=30):
    """Process a single target variable to get related vars and conditions"""
    
    print(f"\nProcessing target variable: {target_var}")
    print("=" * 60)
    
    # Step 1: Get 30 related variables using LLM
    print("Step 1: Getting 30 related variables using LLM...")
    related_vars = get_30_related_variables(generator, target_var, eligible_vars, mappings)
    
    if len(related_vars) < 5:
        return {
            "target_variable": target_var,
            "related_variables": related_vars,
            "selected_constraints": None,
            "sample_count": 0,
            "status": f"Failed: Only {len(related_vars)} related variables found (need 5)"
        }
    
    # Step 2: Find best 5-variable combination
    print("Step 2: Finding best 5-variable combination...")
    best_vars, best_conditions, sample_count = find_best_5_variable_combination(
        df, target_var, related_vars, min_samples
    )
    
    result = {
        "target_variable": target_var,
        "related_variables": related_vars,
        "selected_constraints": best_conditions,
        "constraint_variables": best_vars,
        "sample_count": sample_count
    }
    
    if best_vars:
        result["status"] = "Success"
        constraint_count = len(best_conditions) if best_conditions else 0
        print(f"✓ SUCCESS: Found {constraint_count}-variable combination with {sample_count} samples")
    else:
        result["status"] = "Failed: No valid 5 or 4-variable combination found"
        print("✗ FAILED: No valid 5 or 4-variable combination found")
    
    return result

def generate_smart_conditions(gss_file, mapping_file, output_file, api_key, min_samples=30, target_vars=None, max_variables=None):
    """Main function to generate smart conditions for all eligible variables"""
    
    print("GSS Smart Conditions Generator with LLM Selection")
    print("=" * 60)
    print(f"Input mapping file: {mapping_file}")
    print(f"Output file: {output_file}")
    print(f"Minimum samples required: {min_samples}")
    if max_variables:
        print(f"Processing first {max_variables} variables only")
    print()
    
    # Initialize components
    generator = GSSSmartConditionsGenerator(api_key)
    df = load_gss_dataset(gss_file)
    mappings = load_mappings(mapping_file)
    
    # Update incomplete domains from GSS data
    mappings = update_incomplete_domains(df, mappings)
    print()
    
    # Create output mappings (copy of updated mappings)
    output_mappings = mappings.copy()
    
    # Get eligible variables (for processing as targets)
    eligible_vars = get_eligible_variables(mappings)
    
    # Get all condition candidates (including already processed variables)
    all_condition_candidates = get_all_condition_candidates(mappings)
    
    if len(eligible_vars) == 0:
        print("✓ All eligible variables already have relevant_constraints. Processing complete!")
        return mappings, {"total": 0, "successful": 0, "failed": 0}
    
    if len(all_condition_candidates) < 5:
        print(f"❌ Need at least 5 condition candidates (have {len(all_condition_candidates)})")
        sys.exit(1)
    
    # Determine target variables to process
    if target_vars:
        # Filter to only eligible target vars
        target_vars = [v for v in target_vars if v in eligible_vars]
        print(f"Processing specified target variables: {target_vars}")
    else:
        # Use all eligible variables, but potentially limit the number processed
        target_vars = eligible_vars.copy()
        if max_variables and max_variables < len(target_vars):
            target_vars = target_vars[:max_variables]
            print(f"Processing first {len(target_vars)} of {len(eligible_vars)} eligible variables")
        else:
            print(f"Processing all {len(target_vars)} eligible variables as targets")
    
    print(f"All {len(all_condition_candidates)} eligible variables will be considered as potential conditions")
    
    # Process each target variable
    stats = {"total": 0, "successful": 0, "failed": 0}
    
    for i, target_var in enumerate(target_vars):
        stats["total"] += 1
        print(f"\n[{i+1}/{len(target_vars)}] Processing {target_var}...")
        
        try:
            result = process_single_variable(
                generator, df, target_var, all_condition_candidates, mappings, min_samples
            )
            
            # Add smart conditions to the output mappings
            if result.get("status") == "Success":
                output_mappings[target_var]['related_variables'] = result['related_variables']
                output_mappings[target_var]['relevant_constraints'] = result['selected_constraints']
                output_mappings[target_var]['relevant_sample_count'] = result['sample_count']
                # Remove failure markers if this was a retry
                if 'condition_generation_failed' in output_mappings[target_var]:
                    del output_mappings[target_var]['condition_generation_failed']
                stats["successful"] += 1
            else:
                output_mappings[target_var]['related_variables'] = result.get('related_variables', [])
                output_mappings[target_var]['relevant_constraints'] = None
                output_mappings[target_var]['condition_generation_failed'] = result.get('status', 'Unknown error')
                stats["failed"] += 1
                
        except Exception as e:
            print(f"❌ Error processing {target_var}: {e}")
            output_mappings[target_var]['related_variables'] = []
            output_mappings[target_var]['relevant_constraints'] = None
            output_mappings[target_var]['condition_generation_failed'] = f"Error: {str(e)}"
            stats["failed"] += 1
        
        # Save progress every 5 variables
        if stats["total"] % 5 == 0:
            save_results(output_mappings, output_file, stats)
    
    # Final save
    save_results(output_mappings, output_file, stats, final=True)
    return output_mappings, stats

def save_results(output_mappings, output_file, stats, final=False):
    """Save results to output file in the same format as input mappings"""
    try:
        output_path = Path(output_file)
        if output_path.suffix.lower() != '.json':
            output_path = output_path.with_suffix('.json')
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_mappings, f, indent=2, ensure_ascii=False)
        
        status = "Final save" if final else "Progress save"
        print(f"  {status}: {output_path}")
        if final:
            print(f"  Added smart conditions to {stats['successful']} variables")
        
    except Exception as e:
        print(f"  ❌ Save error: {e}")

def load_api_key():
    """Load OpenAI API key from environment variable or file"""
    api_key = os.getenv('OPENAI_API_KEY')
    if api_key:
        return api_key
    
    api_key_files = ['openai_api_key.txt', 'api_key.txt', '.env']
    for filename in api_key_files:
        if os.path.exists(filename):
            try:
                with open(filename, 'r') as f:
                    content = f.read().strip()
                    if '=' in content:
                        for line in content.split('\n'):
                            if line.startswith('OPENAI_API_KEY='):
                                return line.split('=', 1)[1].strip().strip('"\'')
                    else:
                        return content
            except Exception:
                continue
    
    return None

def test_single_target(gss_file, mapping_file, target_var, api_key, min_samples=30):
    """Test the smart condition generation for a single variable"""
    
    print(f"Testing smart condition generation for: {target_var}")
    print("=" * 50)
    
    generator = GSSSmartConditionsGenerator(api_key)
    df = load_gss_dataset(gss_file)
    mappings = load_mappings(mapping_file)
    eligible_vars = get_eligible_variables(mappings)
    all_condition_candidates = get_all_condition_candidates(mappings)
    
    if target_var not in eligible_vars:
        if target_var in mappings and 'relevant_constraints' in mappings[target_var] and mappings[target_var]['relevant_constraints'] is not None:
            print(f"✓ {target_var} already has relevant_constraints - skipping")
        else:
            print(f"❌ {target_var} is not eligible (must be accepted and not excluded)")
        return
    
    result = process_single_variable(generator, df, target_var, all_condition_candidates, mappings, min_samples)
    
    print(f"\n=== RESULTS FOR {target_var} ===")
    print(f"Status: {result['status']}")
    print(f"Related variables found: {len(result['related_variables'])}")
    
    if result['related_variables']:
        print("Related variables (in order):")
        for i, var in enumerate(result['related_variables']):
            question = mappings[var].get('revised_question', 'No description')
            print(f"  {i+1:2d}. {var}: {question[:60]}...")
    
    if result['selected_constraints']:
        print(f"\nSelected constraints (sample count: {result['sample_count']}):")
        for var, value in result['selected_constraints'].items():
            question = mappings[var].get('revised_question', 'No description')
            print(f"  {var} = '{value}' ({question[:40]}...)")

def main():
    """Main function with command line interface"""
    
    if len(sys.argv) < 4:
        print("Usage: python gss_smart_conditions_generator.py <gss_dataset_file> <mapping_file> <output_file> [options]")
        print("       python gss_smart_conditions_generator.py --test <gss_dataset_file> <mapping_file> <target_variable>")
        print()
        print("Arguments:")
        print("  gss_dataset_file : Path to the local GSS dataset file")
        print("  mapping_file     : Path to the mappings JSON file with revised questions")
        print("  output_file      : Path for the output file with smart conditions")
        print()
        print("Options:")
        print("  --min-samples N  : Minimum samples required (default: 30)")
        print("  --targets VAR1,VAR2 : Process only specified target variables")
        print("  --max-vars N     : Process only the first N variables")
        print()
        print("Test mode:")
        print("  --test target_var : Test smart condition generation for one variable")
        print()
        print("Examples:")
        print("Examples:")
        print("  python gss_smart_conditions_generator.py GSS_2022.csv mappings.json output.json")
        print("  python gss_smart_conditions_generator.py GSS_2022.csv mappings.json output.json --min-samples 25")
        print("  python gss_smart_conditions_generator.py GSS_2022.csv mappings.json output.json --max-vars 10")
        print("  python gss_smart_conditions_generator.py --test GSS_2022.csv mappings.json sex")
        print()
        print("Note: Variables with existing relevant_constraints are automatically skipped for incremental processing.")
        sys.exit(1)
    
    # Load API key
    api_key = load_api_key()
    if not api_key:
        print("❌ OpenAI API key not found!")
        print("Set OPENAI_API_KEY environment variable or create openai_api_key.txt file")
        sys.exit(1)
    
    # Check for test mode
    if sys.argv[1] == "--test":
        if len(sys.argv) != 5:
            print("❌ Test mode requires: --test <gss_file> <mapping_file> <target_variable>")
            sys.exit(1)
        
        gss_file = sys.argv[2]
        mapping_file = sys.argv[3]
        target_var = sys.argv[4]
        
        test_single_target(gss_file, mapping_file, target_var, api_key)
        return
    
    # Normal mode
    gss_file = sys.argv[1]
    mapping_file = sys.argv[2]
    output_file = sys.argv[3]
    
    # Parse options
    min_samples = 20
    target_vars = None
    max_variables = 200
    
    i = 4
    while i < len(sys.argv):
        if sys.argv[i] == "--min-samples" and i + 1 < len(sys.argv):
            min_samples = int(sys.argv[i + 1])
            i += 2
        elif sys.argv[i] == "--targets" and i + 1 < len(sys.argv):
            target_vars = [v.strip() for v in sys.argv[i + 1].split(',')]
            i += 2
        elif sys.argv[i] == "--max-vars" and i + 1 < len(sys.argv):
            max_variables = int(sys.argv[i + 1])
            i += 2
        else:
            print(f"Unknown option: {sys.argv[i]}")
            sys.exit(1)
    
    # Validate input files
    for file_path in [gss_file, mapping_file]:
        if not os.path.exists(file_path):
            print(f"❌ File not found: {file_path}")
            sys.exit(1)
    
    # Run the smart condition generation
    try:
        output_mappings, stats = generate_smart_conditions(
            gss_file, mapping_file, output_file, api_key, min_samples, target_vars, max_variables
        )
        
        print(f"\n🎉 SMART CONDITIONS GENERATION COMPLETE!")
        print(f"Total processed: {stats['total']}")
        print(f"Successful: {stats['successful']}")
        print(f"Failed: {stats['failed']}")
        
        if stats['total'] > 0:
            success_rate = stats['successful'] / stats['total'] * 100
            print(f"Success rate: {success_rate:.1f}%")
        
        print(f"\nOutput saved to: {output_file}")
        print("Added new fields to each processed variable:")
        print("  - related_variables: List of 30 LLM-selected related variables")
        print("  - relevant_constraints: Dictionary of 5 (or 4 fallback) constraint variable-value pairs")
        print("  - relevant_sample_count: Number of valid samples with constraints")
        
    except KeyboardInterrupt:
        print("\n⚠️ Process interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error during processing: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()