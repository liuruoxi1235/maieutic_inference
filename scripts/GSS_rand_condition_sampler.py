import pandas as pd
import json
import sys
import os
import random
from pathlib import Path
import numpy as np

def load_gss_dataset(file_path):
    """
    Load the GSS dataset from various possible formats
    """
    print(f"Loading GSS dataset from: {file_path}")
    
    # Get file extension to determine format
    file_ext = Path(file_path).suffix.lower()
    
    try:
        if file_ext == '.csv':
            # Try different encodings commonly used for GSS data
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
            # Default to CSV
            df = pd.read_csv(file_path, low_memory=False)
            print("✓ Successfully loaded as CSV (default)")
        
        # Convert column names to lowercase for consistency
        df.columns = df.columns.str.lower()
        
        print(f"Dataset shape: {df.shape}")
        print(f"Sample columns: {list(df.columns[:10])}{'...' if len(df.columns) > 10 else ''}")
        
        return df
        
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        print("Supported formats: .csv, .dta, .sav, .xlsx, .xls")
        sys.exit(1)

def load_mappings(mapping_file):
    """
    Load the variable mappings with revised questions
    """
    print(f"Loading mappings from: {mapping_file}")
    
    try:
        with open(mapping_file, 'r', encoding='utf-8') as f:
            mappings = json.load(f)
        
        print(f"✓ Loaded {len(mappings)} variables from mapping file")
        return mappings
        
    except Exception as e:
        print(f"❌ Error loading mappings: {e}")
        sys.exit(1)

def get_accepted_variables(mappings):
    """
    Extract variables that have revision_status = 'accepted'
    Excludes variables with 'ethregion', 'ethworld', or 'raceacs' in their names
    """
    excluded_patterns = ['ethregion', 'ethworld', 'raceacs']
    accepted_vars = []
    
    for var_name, var_info in mappings.items():
        if var_info.get('revision_status') == 'accepted':
            # Check if variable name contains any excluded patterns
            if not any(pattern in var_name.lower() for pattern in excluded_patterns):
                accepted_vars.append(var_name)
    
    print(f"Found {len(accepted_vars)} accepted variables (after excluding ethregion/ethworld/raceacs patterns)")
    return accepted_vars

def get_variable_valid_values(df, var_name, year_filter=2022):
    """
    Get valid values for a variable (excluding common invalid responses)
    Only considers data from the specified year.
    """
    invalid_responses = {
        'not available in this year',
        'not available this year', 
        'not avail in this year',
        'not avail this year',
        'nay',
        'na',
        'n/a',
        'iap',
        'dk',
        "don't know",
        'no answer',
        'skipped on web',
        'refused',
        '.'
    }
    
    if var_name not in df.columns:
        return []
    
    # Filter for specified year first
    year_df = df[df['year'] == year_filter] if 'year' in df.columns else df
    
    # Get unique values excluding NaN and invalid responses
    all_values = year_df[var_name].dropna().unique()
    valid_values = []
    
    for val in all_values:
        val_str = str(val).lower().strip()
        if val_str not in invalid_responses:
            valid_values.append(val)
    
    return list(valid_values)

def test_conditions(df, target_var, condition_vars, condition_values, min_samples=10, verbose=False):
    """
    Test if applying the given conditions leaves at least min_samples valid responses
    for the target variable. Only considers 2022 data.
    """
    # Start with 2022 data only
    filtered_df = df[df['year'] == 2022].copy()
    
    if verbose:
        print(f"      Starting with {len(filtered_df)} rows from 2022")
    
    # Apply each condition
    for cond_var, cond_val in zip(condition_vars, condition_values):
        if cond_var in filtered_df.columns:
            before_count = len(filtered_df)
            filtered_df = filtered_df[filtered_df[cond_var] == cond_val]
            after_count = len(filtered_df)
            if verbose:
                print(f"      After {cond_var} = '{cond_val}': {after_count} rows (was {before_count})")
        else:
            if verbose:
                print(f"      Warning: {cond_var} not found in dataset")
    
    # Check if target variable has enough valid responses
    if target_var not in filtered_df.columns:
        if verbose:
            print(f"      Target variable {target_var} not found in dataset")
        return False, 0
    
    # Count valid responses for target variable
    invalid_responses = {
        'not available in this year',
        'not available this year', 
        'not avail in this year',
        'not avail this year',
        'nay',
        'na',
        'n/a',
        'iap',
        'dk',
        "don't know",
        'no answer',
        'skipped on web',
        'refused',
        '.'
    }
    
    target_values = filtered_df[target_var].dropna()
    valid_target_values = []
    
    for val in target_values:
        val_str = str(val).lower().strip()
        if val_str not in invalid_responses:
            valid_target_values.append(val)
    
    valid_count = len(valid_target_values)
    
    if verbose:
        total_target_responses = len(target_values)
        print(f"      Target variable '{target_var}': {total_target_responses} total responses, {valid_count} valid responses")
    
    return valid_count >= min_samples, valid_count

def find_best_value_for_variable(df, cond_var, current_filter_df, verbose=False):
    """
    For a given condition variable, find the value that leaves the most samples
    when applied to the current filtered dataset
    """
    valid_values = get_variable_valid_values(df, cond_var, year_filter=2022)
    
    if not valid_values:
        return None, 0
    
    best_value = None
    best_count = 0
    
    if verbose:
        print(f"        Testing {len(valid_values)} values for {cond_var}:")
    
    for value in valid_values:
        # Apply this value and count remaining rows
        test_df = current_filter_df[current_filter_df[cond_var] == value]
        count = len(test_df)
        
        if verbose:
            print(f"          '{value}': {count} rows")
        
        if count > best_count:
            best_count = count
            best_value = value
    
    if verbose:
        print(f"        Best value for {cond_var}: '{best_value}' with {best_count} rows")
    
    return best_value, best_count

def generate_random_conditions(df, target_var, candidate_vars, max_attempts=50, min_samples=10, verbose=False):
    """
    Generate random conditions for a target variable by trying different combinations
    of 5 condition variables. For each selected variable, choose the domain value
    that results in the most samples left. Only considers 2022 data.
    Excludes variables with 'ethregion', 'ethworld', or 'raceacs' in their names.
    
    If no 5-constraint combination yields enough samples, falls back to the best
    4-constraint combination found during the process.
    """
    # Remove target variable from candidates and filter out excluded patterns
    excluded_patterns = ['ethregion', 'ethworld', 'raceacs']
    candidate_vars = [
        v for v in candidate_vars 
        if v != target_var 
        and v in df.columns 
        and not any(pattern in v.lower() for pattern in excluded_patterns)
    ]
    
    if len(candidate_vars) < 5:
        return None, f"Not enough candidate variables (need 5, have {len(candidate_vars)} after filtering)"
    
    # Check if we have 2022 data
    if 'year' not in df.columns:
        return None, "No 'year' column found in dataset"
    
    year_2022_df = df[df['year'] == 2022]
    if len(year_2022_df) == 0:
        return None, "No 2022 data found in dataset"
    
    if verbose:
        print(f"    Working with {len(year_2022_df)} rows from 2022")
        print(f"    Available condition variables: {len(candidate_vars)} (excluding ethregion/ethworld/raceacs)")
    
    # Track the best 4-constraint combination as fallback
    best_4_constraints = None
    best_4_constraints_count = 0
    best_4_constraints_message = ""
    
    for attempt in range(max_attempts):
        if verbose:
            print(f"    Attempt {attempt + 1}/{max_attempts}:")
        
        # Randomly select 5 condition variables
        condition_vars = random.sample(candidate_vars, 5)
        
        if verbose:
            print(f"      Selected variables: {condition_vars}")
        
        # Start with 2022 data
        current_df = year_2022_df.copy()
        condition_values = []
        conditions_dict = {}
        
        # For each condition variable, find the value that maximizes remaining samples
        valid_combination = True
        for i, cond_var in enumerate(condition_vars):
            if cond_var not in current_df.columns:
                if verbose:
                    print(f"      {cond_var}: Not found in dataset")
                valid_combination = False
                break
            
            # Find the best value for this variable given current filtering
            best_value, best_count = find_best_value_for_variable(current_df, cond_var, current_df, verbose)
            
            if best_value is None:
                if verbose:
                    print(f"      {cond_var}: No valid values found")
                valid_combination = False
                break
            
            # Apply this condition
            before_count = len(current_df)
            current_df = current_df[current_df[cond_var] == best_value]
            after_count = len(current_df)
            
            condition_values.append(best_value)
            conditions_dict[cond_var] = best_value
            
            if verbose:
                print(f"      Applied {cond_var} = '{best_value}': {after_count} rows (was {before_count})")
            
            # Check if this is a good 4-constraint combination (after 4th constraint)
            if i == 3:  # We've applied 4 constraints
                has_enough_4, sample_count_4 = test_conditions(
                    df, target_var, condition_vars[:4], condition_values[:4], min_samples, verbose=False
                )
                
                if sample_count_4 > best_4_constraints_count:
                    best_4_constraints = {var: val for var, val in zip(condition_vars[:4], condition_values[:4])}
                    best_4_constraints_count = sample_count_4
                    best_4_constraints_message = f"Best 4-constraint fallback with {sample_count_4} samples"
                    
                    if verbose:
                        print(f"      📝 New best 4-constraint combination: {sample_count_4} samples")
            
            # If we have too few rows already, no point continuing
            if after_count == 0:
                if verbose:
                    print(f"      No rows left after applying {cond_var}")
                valid_combination = False
                break
        
        if not valid_combination:
            if verbose:
                print("      ✗ Invalid combination, trying next...")
            continue
        
        # Now test the target variable with all 5 conditions
        has_enough, sample_count = test_conditions(
            df, target_var, condition_vars, condition_values, min_samples, verbose=verbose
        )
        
        if verbose:
            result_status = "✓" if has_enough else "✗"
            print(f"      {result_status} Final result: {sample_count} valid samples for '{target_var}' (need {min_samples})")
        
        if has_enough:
            return conditions_dict, f"Found valid 5-constraint conditions with {sample_count} samples"
        
        if verbose:
            print("      Not enough valid target samples, trying next combination...")
    
    # If we get here, no 5-constraint combination worked
    # Check if we have a good 4-constraint fallback
    if best_4_constraints and best_4_constraints_count >= min_samples:
        if verbose:
            print(f"    💡 Using best 4-constraint fallback: {best_4_constraints_count} samples")
        return best_4_constraints, f"Used 4-constraint fallback with {best_4_constraints_count} samples"
    elif best_4_constraints:
        if verbose:
            print(f"    💡 Using best 4-constraint fallback even though it has only {best_4_constraints_count} samples")
        return best_4_constraints, f"Used 4-constraint fallback with {best_4_constraints_count} samples (below minimum)"
    
    return None, f"Could not find valid conditions after {max_attempts} attempts (best 4-constraint had {best_4_constraints_count} samples)"

def generate_conditions_for_all_variables(gss_file, mapping_file, output_file, min_samples=10, max_attempts=50):
    """
    Generate random conditions for all accepted variables in the mapping file.
    Writes results to a new output file, leaving the original mapping file unchanged.
    """
    print("GSS Random Conditions Generator")
    print("=" * 50)
    print(f"Input mapping file: {mapping_file}")
    print(f"Output file: {output_file}")
    print(f"Minimum samples required: {min_samples}")
    print(f"Maximum attempts per variable: {max_attempts}")
    print()
    
    # Load dataset and mappings
    df = load_gss_dataset(gss_file)
    mappings = load_mappings(mapping_file)
    
    # Create a copy of mappings for output (don't modify original)
    output_mappings = mappings.copy()
    
    # Get accepted variables
    accepted_vars = get_accepted_variables(mappings)
    
    if len(accepted_vars) < 6:
        print(f"❌ Need at least 6 accepted variables to generate conditions (have {len(accepted_vars)})")
        sys.exit(1)
    
    print(f"Generating conditions for {len(accepted_vars)} accepted variables...")
    print("-" * 60)
    
    # Track statistics
    stats = {
        'total_processed': 0,
        'successful': 0,
        'failed': 0
    }
    
    # Process each accepted variable
    for i, target_var in enumerate(accepted_vars):
        stats['total_processed'] += 1
        
        print(f"[{i+1:3d}/{len(accepted_vars)}] Processing {target_var}...")
        
        # Generate conditions for this variable
        conditions, message = generate_random_conditions(
            df, target_var, accepted_vars, max_attempts, min_samples, verbose=True
        )
        
        if conditions:
            # Add conditions to the OUTPUT variable info (not the original)
            output_mappings[target_var]['rand_conditions'] = conditions
            stats['successful'] += 1
            
            # Show the conditions
            print(f"    ✓ SUCCESS: {message}")
            for cond_var, cond_val in conditions.items():
                print(f"      {cond_var} = {cond_val}")
        else:
            # Mark as failed in OUTPUT mappings
            output_mappings[target_var]['rand_conditions'] = None
            output_mappings[target_var]['condition_generation_failed'] = message
            stats['failed'] += 1
            print(f"    ✗ FAILED: {message}")
        
        # Save progress every 10 variables
        if stats['total_processed'] % 10 == 0:
            save_progress(output_mappings, output_file, stats)
    
    # Final save
    save_progress(output_mappings, output_file, stats, final=True)
    
    return stats

def save_progress(output_mappings, output_file, stats, final=False):
    """
    Save current progress to the output file
    """
    try:
        output_path = Path(output_file)
        if output_path.suffix.lower() != '.json':
            output_path = output_path.with_suffix('.json')
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_mappings, f, indent=2, ensure_ascii=False)
        
        status = "Final save" if final else "Progress save"
        print(f"    {status}: {output_path}")
        
    except Exception as e:
        print(f"    ❌ Save error: {e}")

def test_single_variable(gss_file, mapping_file, target_var, min_samples=10):
    """
    Test condition generation for a single variable (for debugging)
    """
    print(f"Testing condition generation for variable: {target_var}")
    print("=" * 50)
    
    # Check if target variable should be excluded
    excluded_patterns = ['ethregion', 'ethworld', 'raceacs']
    if any(pattern in target_var.lower() for pattern in excluded_patterns):
        print(f"❌ Variable {target_var} contains excluded pattern (ethregion/ethworld/raceacs)")
        return
    
    # Load dataset and mappings
    df = load_gss_dataset(gss_file)
    mappings = load_mappings(mapping_file)
    
    # Get accepted variables
    accepted_vars = get_accepted_variables(mappings)
    
    if target_var not in accepted_vars:
        print(f"❌ Variable {target_var} is not in accepted variables")
        # Check if it's because of revision status or exclusion pattern
        if target_var in mappings:
            status = mappings[target_var].get('revision_status', 'unknown')
            print(f"   Revision status: {status}")
        return
    
    # Test the target variable
    print(f"Target variable: {target_var}")
    
    if target_var in mappings:
        var_info = mappings[target_var]
        print(f"Revised question: {var_info.get('revised_question', 'N/A')}")
        print(f"Data type: {var_info.get('data_type', 'N/A')}")
        print(f"Domain values: {var_info.get('domain_values', [])}")
    
    print(f"\nGenerating conditions...")
    
    conditions, message = generate_random_conditions(
        df, target_var, accepted_vars, max_attempts=10, min_samples=min_samples, verbose=True
    )
    
    if conditions:
        print(f"✓ SUCCESS: {message}")
        print("Generated conditions:")
        for cond_var, cond_val in conditions.items():
            print(f"  {cond_var} = {cond_val}")
            
        # Show what the condition variable questions are
        print("\nCondition variable details:")
        for cond_var in conditions.keys():
            if cond_var in mappings:
                cond_info = mappings[cond_var]
                question = cond_info.get('revised_question', 'N/A')
                print(f"  {cond_var}: {question}")
    else:
        print(f"✗ FAILED: {message}")

def main():
    """
    Main function with command line interface
    """
    if len(sys.argv) < 4:
        print("Usage: python gss_conditions_generator.py <gss_dataset_file> <mapping_file> <output_file> [options]")
        print("       python gss_conditions_generator.py --test <gss_dataset_file> <mapping_file> <target_variable>")
        print()
        print("Arguments:")
        print("  gss_dataset_file : Path to the local GSS dataset file (.csv, .dta, .sav, .xlsx)")
        print("  mapping_file     : Path to the input mappings JSON file with revised questions")
        print("  output_file      : Path for the NEW output file with conditions (will not modify input)")
        print()
        print("Options:")
        print("  --min-samples N  : Minimum samples required (default: 10)")
        print("  --max-attempts N : Maximum attempts per variable (default: 50)")
        print()
        print("Test mode:")
        print("  --test           : Test condition generation for a single variable")
        print("  target_variable  : Variable name to test (in test mode)")
        print()
        print("Examples:")
        print("  python gss_conditions_generator.py GSS_2022.csv gss2022_revised.json gss2022_with_conditions.json")
        print("  python gss_conditions_generator.py GSS_2022.csv gss2022_revised.json output.json --min-samples 15")
        print("  python gss_conditions_generator.py --test GSS_2022.csv gss2022_revised.json sex")
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
    output_file = sys.argv[3]
    
    # Parse options
    min_samples = 10
    max_attempts = 50
    
    i = 4
    while i < len(sys.argv):
        if sys.argv[i] == "--min-samples" and i + 1 < len(sys.argv):
            min_samples = int(sys.argv[i + 1])
            i += 2
        elif sys.argv[i] == "--max-attempts" and i + 1 < len(sys.argv):
            max_attempts = int(sys.argv[i + 1])
            i += 2
        else:
            print(f"Unknown option: {sys.argv[i]}")
            sys.exit(1)
    
    # Validate input files
    if not os.path.exists(gss_file):
        print(f"❌ GSS dataset file not found: {gss_file}")
        sys.exit(1)
        
    if not os.path.exists(mapping_file):
        print(f"❌ Mapping file not found: {mapping_file}")
        sys.exit(1)
    
    # Generate conditions
    try:
        stats = generate_conditions_for_all_variables(
            gss_file, mapping_file, output_file, min_samples, max_attempts
        )
        
        # Print final summary
        print(f"\n🎉 CONDITION GENERATION COMPLETE!")
        print(f"Total processed: {stats['total_processed']}")
        print(f"Successful: {stats['successful']}")
        print(f"Failed: {stats['failed']}")
        
        if stats['total_processed'] > 0:
            success_rate = stats['successful'] / stats['total_processed'] * 100
            print(f"Success rate: {success_rate:.1f}%")
        
        print(f"\nOutput saved to: {output_file}")
        
    except KeyboardInterrupt:
        print("\n⚠️ Process interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error during processing: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()