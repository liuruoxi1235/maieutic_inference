import json
import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np

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

def get_variables_with_prompts(mappings):
    """Get variables that have existing prompts (rand_prompts or relevant_prompts)"""
    valid_vars = []
    
    for var_name, var_info in mappings.items():
        has_rand_prompts = 'rand_prompts' in var_info and var_info['rand_prompts'] is not None and len(var_info['rand_prompts']) > 0
        has_relevant_prompts = 'relevant_prompts' in var_info and var_info['relevant_prompts'] is not None and len(var_info['relevant_prompts']) > 0
        
        # Skip if both are present (shouldn't happen) or neither are present
        if (has_rand_prompts and has_relevant_prompts) or (not has_rand_prompts and not has_relevant_prompts):
            continue
            
        valid_vars.append(var_name)
    
    print(f"Found {len(valid_vars)} variables with existing prompts to process for branching")
    return valid_vars

def get_answer_distribution(df, target_var, conditions_dict, target_domain_values):
    """
    Get the answer distribution for target_var given conditions from the 2022 GSS dataset
    """
    print(f"        🔍 Debug: get_answer_distribution called")
    print(f"        🔍 Debug:   target_var = {target_var}")
    print(f"        🔍 Debug:   conditions_dict = {conditions_dict}")
    print(f"        🔍 Debug:   target_domain_values = {target_domain_values}")
    
    # Filter for 2022 data
    df_2022 = df[df['year'] == 2022] if 'year' in df.columns else df
    print(f"        🔍 Debug: df_2022 shape = {df_2022.shape}")
    
    if len(df_2022) == 0:
        print(f"        🔍 Debug: ✗ No 2022 data found")
        return None
    
    # Apply conditions
    filtered_df = df_2022.copy()
    print(f"        🔍 Debug: Starting with {len(filtered_df)} rows")
    
    for var, value in conditions_dict.items():
        print(f"        🔍 Debug: Applying condition {var} = {value}")
        if var in filtered_df.columns:
            before_count = len(filtered_df)
            filtered_df = filtered_df[filtered_df[var] == value]
            after_count = len(filtered_df)
            print(f"        🔍 Debug:   {var} in columns, filtered from {before_count} to {after_count} rows")
        else:
            print(f"        🔍 Debug:   ✗ {var} NOT in columns")
            print(f"        🔍 Debug:   Available columns sample: {list(filtered_df.columns)[:10]}...")
    
    print(f"        🔍 Debug: After all conditions: {len(filtered_df)} rows")
    
    if len(filtered_df) == 0:
        print(f"        🔍 Debug: ✗ No rows after applying conditions")
        return None
    
    # Get target variable responses
    if target_var not in filtered_df.columns:
        print(f"        🔍 Debug: ✗ target_var '{target_var}' not in columns")
        print(f"        🔍 Debug:   Available columns sample: {list(filtered_df.columns)[:10]}...")
        return None
    
    target_responses = filtered_df[target_var].dropna()
    print(f"        🔍 Debug: target_responses count = {len(target_responses)}")
    print(f"        🔍 Debug: target_responses unique values = {target_responses.unique()}")
    
    # Invalid responses to exclude
    invalid_responses = {
        'not available in this year', 'not available this year', 'not avail in this year',
        'not avail this year', 'nay', 'na', 'n/a', 'iap', 'dk', "don't know",
        'no answer', 'skipped on web', 'refused', '.'
    }
    
    # Filter out invalid responses
    valid_responses = []
    for response in target_responses:
        if str(response).lower().strip() not in invalid_responses:
            valid_responses.append(response)
    
    print(f"        🔍 Debug: valid_responses count = {len(valid_responses)}")
    print(f"        🔍 Debug: valid_responses sample = {valid_responses[:10]}")
    
    if len(valid_responses) == 0:
        print(f"        🔍 Debug: ✗ No valid responses after filtering")
        return None
    
    # Count occurrences of each domain value
    domain_counts = {}
    for domain_val in target_domain_values:
        domain_counts[domain_val] = 0
    
    # Count valid responses
    matched_count = 0
    for response in valid_responses:
        if response in target_domain_values:
            domain_counts[response] += 1
            matched_count += 1
    
    print(f"        🔍 Debug: domain_counts = {domain_counts}")
    print(f"        🔍 Debug: matched_count = {matched_count} / {len(valid_responses)}")
    
    total_valid = sum(domain_counts.values())
    if total_valid == 0:
        print(f"        🔍 Debug: ✗ No responses matched domain values")
        return None
    
    # Convert to probabilities
    probabilities = []
    for domain_val in target_domain_values:
        prob = domain_counts[domain_val] / total_valid
        probabilities.append(round(prob, 4))
    
    print(f"        🔍 Debug: ✓ probabilities = {probabilities}")
    return probabilities

def get_variable_domain_values(mappings, var_name):
    """Get domain values for a variable from mappings"""
    if var_name not in mappings:
        return None
    
    var_info = mappings[var_name]
    all_domain_values = var_info.get('domain_values', [])
    
    # Filter out invalid response types
    invalid_responses = {
        'not available in this year', 'not available this year', 'not avail in this year',
        'not avail this year', 'nay', 'na', 'n/a', 'iap', 'dk', "don't know",
        'no answer', 'skipped on web', 'refused', '.'
    }
    
    valid_domain = []
    for val in all_domain_values:
        if str(val).lower().strip() not in invalid_responses:
            valid_domain.append(val)
    
    return valid_domain if valid_domain else None

def get_variable_question(mappings, var_name):
    """Get the revised question for a variable from mappings"""
    if var_name not in mappings:
        return None
    
    return mappings[var_name].get('revised_question', '')

def create_facts_string(condition_vars, condition_values, mappings):
    """
    Create the facts string from condition variables and values
    """
    facts = []
    
    for var, value in zip(condition_vars, condition_values):
        question = get_variable_question(mappings, var)
        if question:
            # Create fact statement: "Question? Value."
            fact = f"{question} {value}."
            facts.append(fact)
    
    return ' \n '.join(facts)

def get_prompt_template():
    """Return the fixed prompt template"""
    return [
        ["system", "You are assisting a general social study with your commonsense knowledge. We are interested in the answer distributions of particular questions that were being asked in a 2022 General Social Study in the US. Our ultimate goal is to predict the answer distribution of a specific question from a wide and diverse pool of respondents that participated in that 2022 social study. In order to have more informed answer distribution predictions, we repeatedly ask relevant and helpful sub-questions that are easier to answer, make predictions for the target question answer distribution at each specified case, and finally aggregate those answers for a final prediction. "],
        ["system", "Your specific job is to estimate the answer distribution for a question given some known facts. The question may be the target question, or a sub-question that is helpful for the target question once we know its answer distribution. Make an informed prediction that takes into consideration the list of facts that is provided. Return a list of decimals that sums to 0. Each element in the list should correspond to one value in the domain. Here are some predictions that are accurate: "],
        ["user", "Question: What the highest degree of the mother of the respondent in a 2022 US social study?; [associate/junior college; bachelors; graduate; high school; less than high school] "],
        ["assistant", "[0.0627; 0.143; 0.0755; 0.4971; 0.2217]"],
        ["user", "Question: Is the respondent happy with their current job? [Yes; No] \n\n Facts: \n The respondent is working in the finance industry. \n The respondent has worked for less than three years."],
        ["assistant", "[0.2379; 0.7621]"],
        ["user", "Question: {q} \n\n Facts: \n {f} "]
    ]

def format_probability_list(probabilities):
    """Format probability list as string for completion"""
    if probabilities is None:
        return None
    
    # Format each probability to 4 decimal places
    formatted_probs = [f"{prob:.4f}" for prob in probabilities]
    return "[" + "; ".join(formatted_probs) + "]"

def generate_branching_prompt(df, prompt_data, mappings):
    """
    Generate a branching prompt for a single existing prompt
    Target: last condition variable
    Conditions: all other conditions (n-1)
    """
    print(f"      🔍 Debug: Starting generate_branching_prompt")
    
    # Get conditions used in the original prompt
    if 'conditions_used' not in prompt_data or not prompt_data['conditions_used']:
        print(f"      🔍 Debug: ✗ No conditions_used found or empty")
        return None
    
    conditions_used = prompt_data['conditions_used']
    condition_vars = list(conditions_used.keys())
    condition_values = list(conditions_used.values())
    
    print(f"      🔍 Debug: condition_vars = {condition_vars}")
    print(f"      🔍 Debug: condition_values = {condition_values}")
    
    # Need at least 1 condition to create a branching prompt
    if len(condition_vars) < 1:
        print(f"      🔍 Debug: ✗ Less than 1 condition found")
        return None
    
    # Split: last condition becomes target, rest become conditions
    if len(condition_vars) == 1:
        # Special case: only 1 condition, so no original conditions
        target_var = condition_vars[0]
        original_conditions = {}
        original_vars = []
        original_values = []
        print(f"      🔍 Debug: Single condition case - target_var = {target_var}")
    else:
        target_var = condition_vars[-1]  # Last condition becomes target
        original_vars = condition_vars[:-1]  # Rest become conditions
        original_values = condition_values[:-1]
        original_conditions = dict(zip(original_vars, original_values))
        print(f"      🔍 Debug: Multi condition case - target_var = {target_var}")
        print(f"      🔍 Debug: original_vars = {original_vars}")
        print(f"      🔍 Debug: original_conditions = {original_conditions}")
    
    # Get domain values for the new target variable
    target_domain_values = get_variable_domain_values(mappings, target_var)
    print(f"      🔍 Debug: target_domain_values = {target_domain_values}")
    if not target_domain_values:
        print(f"      🔍 Debug: ✗ No valid domain values for target variable {target_var}")
        return None
    
    # Get question for the new target variable
    target_question = get_variable_question(mappings, target_var)
    print(f"      🔍 Debug: target_question = '{target_question}'")
    if not target_question:
        print(f"      🔍 Debug: ✗ No question found for target variable {target_var}")
        return None
    
    # Format domain values for question
    domain_str = "[" + "; ".join(target_domain_values) + "]"
    question_text = f"{target_question} {domain_str}"
    
    # Create facts string from original conditions
    if original_vars:
        facts_str = create_facts_string(original_vars, original_values, mappings)
        print(f"      🔍 Debug: facts_str = '{facts_str[:100]}...'")
    else:
        facts_str = ""  # No facts for the case with 0 original conditions
        print(f"      🔍 Debug: No original conditions, facts_str is empty")
    
    # Get answer distribution from GSS data
    print(f"      🔍 Debug: Calling get_answer_distribution with:")
    print(f"      🔍 Debug:   target_var = {target_var}")
    print(f"      🔍 Debug:   original_conditions = {original_conditions}")
    print(f"      🔍 Debug:   target_domain_values = {target_domain_values}")
    
    probabilities = get_answer_distribution(df, target_var, original_conditions, target_domain_values)
    print(f"      🔍 Debug: probabilities = {probabilities}")
    
    if probabilities is None:
        print(f"      🔍 Debug: ✗ get_answer_distribution returned None")
        return None
    
    # Create prompt by filling template
    prompt_template = get_prompt_template()
    
    # Fill in the question and facts
    filled_prompt = []
    for role, content in prompt_template:
        if role == "user" and "{q}" in content and "{f}" in content:
            if facts_str:
                content = content.replace("{q}", question_text).replace("{f}", facts_str)
            else:
                # Remove the Facts section if there are no facts
                content = f"Question: {question_text}"
        filled_prompt.append([role, content])
    
    # Create completion
    completion = format_probability_list(probabilities)
    print(f"      🔍 Debug: completion = {completion}")
    
    if not completion:
        print(f"      🔍 Debug: ✗ format_probability_list returned None/empty")
        return None
    
    branching_prompt = {
        "prompt": filled_prompt,
        "completion": completion,
        "target_variable": target_var,
        "original_conditions": original_conditions,
        "num_original_conditions": len(original_vars)
    }
    
    print(f"      🔍 Debug: ✓ Successfully created branching prompt")
    return branching_prompt

def process_variable_prompts(df, var_name, var_info, mappings):
    """
    Process all prompts for a variable to generate branching prompts
    """
    print(f"    🔍 Debug: Processing variable {var_name}")
    
    # Determine which type of prompts this variable has
    has_rand_prompts = 'rand_prompts' in var_info and var_info['rand_prompts'] is not None and len(var_info['rand_prompts']) > 0
    has_relevant_prompts = 'relevant_prompts' in var_info and var_info['relevant_prompts'] is not None and len(var_info['relevant_prompts']) > 0
    
    print(f"    🔍 Debug: has_rand_prompts = {has_rand_prompts}")
    print(f"    🔍 Debug: has_relevant_prompts = {has_relevant_prompts}")
    
    if has_rand_prompts:
        prompts_data = var_info['rand_prompts']
        print(f"    🔍 Debug: Using rand_prompts, found {len(prompts_data)} prompts")
    elif has_relevant_prompts:
        prompts_data = var_info['relevant_prompts']
        print(f"    🔍 Debug: Using relevant_prompts, found {len(prompts_data)} prompts")
    else:
        print(f"    🔍 Debug: No valid prompts found")
        return 0  # No prompts to process
    
    processed_count = 0
    
    for i, prompt_data in enumerate(prompts_data):
        print(f"    🔍 Debug: Processing prompt {i+1}/{len(prompts_data)}")
        
        # Check if branching_prompt already exists and is valid (not null)
        if 'branching_prompt' in prompt_data and prompt_data['branching_prompt'] is not None:
            print(f"    🔍 Debug: Prompt {i+1} already has valid branching_prompt, skipping")
            continue
        
        # If branching_prompt is null or doesn't exist, try to generate it
        if 'branching_prompt' in prompt_data and prompt_data['branching_prompt'] is None:
            print(f"    🔍 Debug: Prompt {i+1} has null branching_prompt, retrying generation")
        else:
            print(f"    🔍 Debug: Prompt {i+1} has no branching_prompt, generating")
        
        print(f"    🔍 Debug: Prompt {i+1} data keys: {list(prompt_data.keys())}")
        print(f"    🔍 Debug: conditions_used: {prompt_data.get('conditions_used', 'NOT FOUND')}")
        print(f"    🔍 Debug: num_conditions: {prompt_data.get('num_conditions', 'NOT FOUND')}")
        
        # Generate branching prompt
        branching_prompt = generate_branching_prompt(df, prompt_data, mappings)
        
        if branching_prompt:
            # Add branching prompt to the original prompt data
            prompt_data['branching_prompt'] = branching_prompt
            # Remove any previous failure message
            if 'branching_generation_failed' in prompt_data:
                del prompt_data['branching_generation_failed']
            processed_count += 1
            print(f"    🔍 Debug: ✓ Successfully generated branching prompt for prompt {i+1}")
        else:
            # Mark as failed
            prompt_data['branching_prompt'] = None
            prompt_data['branching_generation_failed'] = "Could not generate branching prompt"
            print(f"    🔍 Debug: ✗ Failed to generate branching prompt for prompt {i+1}")
    
    print(f"    🔍 Debug: Total processed count: {processed_count}")
    return processed_count

def process_all_variables(gss_file, mapping_file, max_variables=None):
    """
    Process all variables with existing prompts to generate branching prompts
    """
    print("GSS Branching Prompt Generator")
    print("=" * 50)
    print(f"Input/Output mapping file: {mapping_file}")
    if max_variables:
        print(f"Processing maximum {max_variables} variables")
    print()
    
    # Load data
    df = load_gss_dataset(gss_file)
    mappings = load_mappings(mapping_file)
    
    # Get variables with existing prompts
    valid_vars = get_variables_with_prompts(mappings)
    
    if len(valid_vars) == 0:
        print("✓ No variables with existing prompts found. Nothing to process!")
        return {"total_processed": 0, "successful": 0, "failed": 0, "total_branching_prompts": 0}
    
    # Limit to max_variables if specified
    if max_variables and max_variables < len(valid_vars):
        valid_vars = valid_vars[:max_variables]
        print(f"Processing first {len(valid_vars)} variables")
    
    print(f"Processing {len(valid_vars)} variables with existing prompts...")
    print("-" * 50)
    
    # Track statistics
    stats = {
        'total_processed': 0,
        'successful': 0,
        'failed': 0,
        'total_branching_prompts': 0
    }
    
    # Process each variable
    for i, var_name in enumerate(valid_vars):
        stats['total_processed'] += 1
        
        print(f"[{i+1:3d}/{len(valid_vars)}] Processing {var_name}...")
        
        try:
            var_info = mappings[var_name]
            branching_count = process_variable_prompts(df, var_name, var_info, mappings)
            
            if branching_count > 0:
                stats['successful'] += 1
                stats['total_branching_prompts'] += branching_count
                print(f"    ✓ Generated {branching_count} branching prompts")
            else:
                stats['failed'] += 1
                print(f"    ✗ No branching prompts generated")
                
        except Exception as e:
            print(f"    ❌ Error processing {var_name}: {e}")
            stats['failed'] += 1
        
        # Save progress every 10 variables
        if stats['total_processed'] % 10 == 0:
            save_results(mappings, mapping_file, stats)
    
    # Final save
    save_results(mappings, mapping_file, stats, final=True)
    
    return stats

def save_results(mappings, mapping_file, stats, final=False):
    """Save results to mapping file (overwrites input file)"""
    try:
        mapping_path = Path(mapping_file)
        if mapping_path.suffix.lower() != '.json':
            mapping_path = mapping_path.with_suffix('.json')
        
        with open(mapping_path, 'w', encoding='utf-8') as f:
            json.dump(mappings, f, indent=2, ensure_ascii=False)
        
        status = "Final save" if final else "Progress save"
        print(f"  {status}: {mapping_path}")
        if final:
            print(f"  Added branching prompts to {stats['successful']} variables")
            print(f"  Total branching prompts generated: {stats['total_branching_prompts']}")
        
    except Exception as e:
        print(f"  ❌ Save error: {e}")

def test_single_variable(gss_file, mapping_file, target_var):
    """Test branching prompt generation for a single variable"""
    
    print(f"Testing branching prompt generation for: {target_var}")
    print("=" * 50)
    
    # Load data
    df = load_gss_dataset(gss_file)
    mappings = load_mappings(mapping_file)
    
    if target_var not in mappings:
        print(f"❌ Variable {target_var} not found in mappings")
        return
    
    var_info = mappings[target_var]
    
    has_rand_prompts = 'rand_prompts' in var_info and var_info['rand_prompts'] is not None and len(var_info['rand_prompts']) > 0
    has_relevant_prompts = 'relevant_prompts' in var_info and var_info['relevant_prompts'] is not None and len(var_info['relevant_prompts']) > 0
    
    if not has_rand_prompts and not has_relevant_prompts:
        print(f"❌ Variable {target_var} has no existing prompts")
        return
    
    if has_rand_prompts and has_relevant_prompts:
        print(f"❌ Variable {target_var} has both rand_prompts and relevant_prompts")
        return
    
    prompts_type = "rand_prompts" if has_rand_prompts else "relevant_prompts"
    prompts_data = var_info[prompts_type]
    
    print(f"Target variable: {target_var}")
    print(f"Prompts type: {prompts_type}")
    print(f"Number of existing prompts: {len(prompts_data)}")
    print()
    
    # Process each prompt
    for i, prompt_data in enumerate(prompts_data):
        print(f"\n--- Processing Prompt {i+1} ---")
        print(f"Original conditions: {prompt_data.get('conditions_used', {})}")
        print(f"Original num_conditions: {prompt_data.get('num_conditions', 'N/A')}")
        
        # Generate branching prompt
        branching_prompt = generate_branching_prompt(df, prompt_data, mappings)
        
        if branching_prompt:
            print(f"✓ Generated branching prompt")
            print(f"  New target variable: {branching_prompt['target_variable']}")
            print(f"  Original conditions for new prompt: {branching_prompt['original_conditions']}")
            print(f"  Num original conditions: {branching_prompt['num_original_conditions']}")
            print(f"  Completion: {branching_prompt['completion']}")
            
            # Show part of the prompt
            final_prompt = None
            for role, content in branching_prompt['prompt']:
                if role == "user" and ("Question:" in content):
                    final_prompt = content
                    break
            
            if final_prompt:
                print(f"  Prompt excerpt: {final_prompt[:200]}...")
        else:
            print(f"✗ Failed to generate branching prompt")

def main():
    """Main function with command line interface"""
    
    if len(sys.argv) < 3:
        print("Usage: python gss_branching_prompt_generator.py <gss_dataset_file> <mapping_file> [options]")
        print("       python gss_branching_prompt_generator.py --test <gss_dataset_file> <mapping_file> <target_variable>")
        print()
        print("Arguments:")
        print("  gss_dataset_file : Path to the local GSS dataset file")
        print("  mapping_file     : Path to the mappings JSON file with existing prompts (will be modified in-place)")
        print()
        print("Options:")
        print("  --max-vars N     : Process only the first N variables")
        print()
        print("Test mode:")
        print("  --test target_var : Test branching prompt generation for one variable")
        print()
        print("Examples:")
        print("  python gss_branching_prompt_generator.py GSS_2022.csv mappings_with_prompts.json")
        print("  python gss_branching_prompt_generator.py GSS_2022.csv mappings.json --max-vars 10")
        print("  python gss_branching_prompt_generator.py --test GSS_2022.csv mappings_with_prompts.json ptnrornt")
        print()
        print("Note: This script modifies the input mapping file in-place, adding 'branching_prompt' to each existing prompt.")
        sys.exit(1)
    
    # Check for test mode
    if sys.argv[1] == "--test":
        if len(sys.argv) != 5:
            print("❌ Test mode requires: --test <gss_file> <mapping_file> <target_variable>")
            sys.exit(1)
        
        gss_file = sys.argv[2]
        mapping_file = sys.argv[3]
        target_var = sys.argv[4]
        
        test_single_variable(gss_file, mapping_file, target_var)
        return
    
    # Normal mode
    gss_file = sys.argv[1]
    mapping_file = sys.argv[2]
    
    # Parse options
    max_variables = None
    
    i = 3
    while i < len(sys.argv):
        if sys.argv[i] == "--max-vars" and i + 1 < len(sys.argv):
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
    
    # Run the branching prompt generation
    try:
        stats = process_all_variables(gss_file, mapping_file, max_variables)
        
        print(f"\n🎉 BRANCHING PROMPT GENERATION COMPLETE!")
        print(f"Total processed: {stats['total_processed']}")
        print(f"Successful: {stats['successful']}")
        print(f"Failed: {stats['failed']}")
        print(f"Total branching prompts generated: {stats['total_branching_prompts']}")
        
        if stats['total_processed'] > 0:
            success_rate = stats['successful'] / stats['total_processed'] * 100
            avg_branching_per_var = stats['total_branching_prompts'] / stats['successful'] if stats['successful'] > 0 else 0
            print(f"Success rate: {success_rate:.1f}%")
            print(f"Average branching prompts per successful variable: {avg_branching_per_var:.1f}")
        
        print(f"\nModified file: {mapping_file}")
        print("Added new field to each existing prompt:")
        print("  - branching_prompt: Prompt-completion pair predicting P(new_condition | original_conditions)")
        
    except KeyboardInterrupt:
        print("\n⚠️ Process interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error during processing: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()