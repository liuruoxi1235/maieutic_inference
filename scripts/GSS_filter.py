import pandas as pd
import json
import sys
import os
from pathlib import Path

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
        print(f"Columns: {list(df.columns[:10])}{'...' if len(df.columns) > 10 else ''}")
        
        return df
        
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        print("Supported formats: .csv, .dta, .sav, .xlsx, .xls")
        sys.exit(1)

def load_scraped_mappings(mapping_file):
    """
    Load the scraped variable mappings
    """
    print(f"Loading scraped mappings from: {mapping_file}")
    
    try:
        with open(mapping_file, 'r', encoding='utf-8') as f:
            mappings = json.load(f)
        
        print(f"✓ Loaded {len(mappings)} variables from mapping file")
        return mappings
        
    except Exception as e:
        print(f"❌ Error loading mappings: {e}")
        sys.exit(1)

def find_valid_2022_variables(df):
    """
    Efficiently find all variables that have valid data in 2022 by iterating through rows.
    Returns a set of variable names that are available in 2022.
    """
    print("Finding variables with valid 2022 data...")
    
    # Filter for 2022 data first
    df_2022 = df[df['year'] == 2022]
    print(f"Found {len(df_2022)} rows for year 2022")
    
    if len(df_2022) == 0:
        print("❌ No 2022 data found")
        return set()
    
    # Common representations of "not available in this year"
    not_available_values = {
        'not available in this year',
        'not available this year', 
        'not avail in this year',
        'not avail this year',
        'nay',  # Common GSS abbreviation
        'na',
        'n/a',
        'iap',  # Inapplicable 
        'dk',   # Don't know
        '.',    # Missing value indicator
        'nan'   # String representation of NaN
    }
    
    valid_variables = set()
    
    # Iterate through each row of 2022 data
    for idx, row in df_2022.iterrows():
        if idx % 100 == 0:  # Progress indicator
            print(f"  Processing row {idx}...")
        
        # Check each column in this row
        for col_name, value in row.items():
            # Skip the year column itself
            if col_name == 'year':
                continue
                
            # Convert value to string and normalize
            if pd.isna(value):
                continue
                
            value_str = str(value).lower().strip()
            
            # If this value is not in the "not available" list, mark variable as valid
            if value_str not in not_available_values:
                valid_variables.add(col_name)
    
    print(f"✓ Found {len(valid_variables)} variables with valid 2022 data")
    return valid_variables

def filter_variables_for_2022(gss_file, mapping_file, output_file):
    """
    Efficiently filter scraped mappings to keep only variables that are:
    1. Available in 2022 GSS with valid data  
    2. Have discrete data type (not numeric)
    3. Have 8 or fewer domain values
    """
    print("GSS 2022 Variable Filter (Optimized)")
    print("=" * 50)
    print("Filtering criteria:")
    print("  ✓ Available in 2022 GSS with valid data")
    print("  ✓ Discrete data type (not numeric)")
    print("  ✓ 8 or fewer domain values")
    print()
    
    # Load the GSS dataset
    df = load_gss_dataset(gss_file)
    
    # Check if year column exists
    if 'year' not in df.columns:
        print("❌ 'year' column not found in dataset")
        print(f"Available columns: {list(df.columns)}")
        sys.exit(1)
    
    # Check available years
    available_years = sorted(df['year'].unique())
    print(f"Available years in dataset: {available_years}")
    
    if 2022 not in available_years:
        print("❌ 2022 data not found in dataset")
        sys.exit(1)
    
    # Step 1: Efficiently find all variables with valid 2022 data
    valid_2022_variables = find_valid_2022_variables(df)
    
    if not valid_2022_variables:
        print("❌ No variables found with valid 2022 data")
        sys.exit(1)
    
    # Load scraped mappings
    mappings = load_scraped_mappings(mapping_file)
    
    # Step 2: Filter based on mapping criteria (discrete + ≤8 domains)
    print(f"\nFiltering {len(valid_2022_variables)} valid variables from mappings...")
    print("-" * 50)
    
    filtered_mappings = {}
    stats = {
        'valid_in_2022': len(valid_2022_variables),
        'found_in_mappings': 0,
        'suitable': 0,
        'numeric_type': 0,
        'too_many_domains': 0,
        'not_in_mappings': 0
    }
    
    # Check each valid 2022 variable against mapping criteria
    for var_name in valid_2022_variables:
        if var_name in mappings:
            stats['found_in_mappings'] += 1
            var_info = mappings[var_name]
            
            # Check data type and domain constraints
            data_type = var_info.get('data_type', '').lower()
            domain_values = var_info.get('domain_values', [])
            
            # Apply filtering criteria
            if data_type == 'numeric':
                stats['numeric_type'] += 1
                reason = "Numeric data type (excluded)"
                status = "✗ EXCLUDED"
            elif data_type == 'discrete' and len(domain_values) > 8:
                stats['too_many_domains'] += 1
                reason = f"Too many domain values ({len(domain_values)} > 8)"
                status = "✗ EXCLUDED"
            else:
                # Variable meets all criteria
                filtered_mappings[var_name] = var_info
                stats['suitable'] += 1
                reason = f"Suitable: discrete, {len(domain_values)} domain values"
                status = "✓ INCLUDED"
            
            # Print progress for key variables
            if var_name.lower() in ['age', 'sex', 'race', 'educ', 'income', 'marital', 'class', 'polviews']:
                domain_count = len(domain_values)
                print(f"{var_name:12} | {data_type:8} | {domain_count:2}d | {status:12} | {reason}")
                
        else:
            stats['not_in_mappings'] += 1
    
    # Save filtered mappings
    print(f"\nSaving filtered mappings to: {output_file}")
    
    # Determine output format based on extension
    output_path = Path(output_file)
    
    if output_path.suffix.lower() == '.json':
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(filtered_mappings, f, indent=2, ensure_ascii=False)
    else:
        # Also save as JSON with .json extension
        json_file = output_path.with_suffix('.json')
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(filtered_mappings, f, indent=2, ensure_ascii=False)
        output_file = str(json_file)
    
    print(f"✓ Saved {len(filtered_mappings)} filtered variables")
    
    # Print summary statistics
    print(f"\n=== OPTIMIZED FILTERING SUMMARY ===")
    print(f"Variables with valid 2022 data: {stats['valid_in_2022']}")
    print(f"Found in mapping file: {stats['found_in_mappings']}")
    print(f"Variables meeting all criteria: {stats['suitable']}")
    print(f"Excluded - Numeric data type: {stats['numeric_type']}")
    print(f"Excluded - Too many domains (>8): {stats['too_many_domains']}")
    print(f"Not found in mappings: {stats['not_in_mappings']}")
    
    if stats['found_in_mappings'] > 0:
        mapping_retention = stats['suitable'] / stats['found_in_mappings'] * 100
        print(f"Mapping retention rate: {mapping_retention:.1f}%")
    
    overall_retention = stats['suitable'] / len(mappings) * 100 if mappings else 0
    print(f"Overall retention rate: {overall_retention:.1f}%")
    
    # Show some examples of filtered variables
    print(f"\n=== SAMPLE SUITABLE VARIABLES ===")
    for i, (var_name, var_info) in enumerate(filtered_mappings.items()):
        if i < 5:
            question = var_info.get('question', 'No description')[:50]
            data_type = var_info.get('data_type', 'unknown')
            domain_count = len(var_info.get('domain_values', []))
            domain_preview = ', '.join(var_info.get('domain_values', [])[:3])
            if len(var_info.get('domain_values', [])) > 3:
                domain_preview += '...'
            print(f"{var_name:12} | {data_type:8} | {domain_count}d | {question}...")
            print(f"{'':12} | {'':8} | {'':2}  | Domains: {domain_preview}")
        else:
            break
    
    if len(filtered_mappings) > 5:
        print(f"... and {len(filtered_mappings) - 5} more variables")
    
    return len(filtered_mappings)

def main():
    """
    Main function with command line interface
    """
    if len(sys.argv) != 4:
        print("Usage: python gss_filter_2022.py <gss_dataset_file> <mapping_file> <output_file>")
        print()
        print("Arguments:")
        print("  gss_dataset_file  : Path to the local GSS dataset file (.csv, .dta, .sav, .xlsx)")
        print("  mapping_file      : Path to the scraped mappings JSON file")
        print("  output_file       : Path for the filtered output file (.json)")
        print()
        print("Filtering criteria:")
        print("  - Available in 2022 GSS with valid data")
        print("  - Discrete data type (not numeric)")
        print("  - 8 or fewer domain values")
        print()
        print("Example:")
        print("  python gss_filter_2022.py GSS_2022.csv gss2024_complete.json gss2022_filtered.json")
        sys.exit(1)
    
    gss_file = sys.argv[1]
    mapping_file = sys.argv[2] 
    output_file = sys.argv[3]
    
    # Validate input files exist
    if not os.path.exists(gss_file):
        print(f"❌ GSS dataset file not found: {gss_file}")
        sys.exit(1)
        
    if not os.path.exists(mapping_file):
        print(f"❌ Mapping file not found: {mapping_file}")
        sys.exit(1)
    
    # Run the filtering
    try:
        num_filtered = filter_variables_for_2022(gss_file, mapping_file, output_file)
        print(f"\n🎉 Successfully filtered to {num_filtered} variables meeting all criteria!")
        print("Variables kept are: discrete type, ≤8 domain values, available in 2022")
        
    except KeyboardInterrupt:
        print("\n⚠️ Process interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error during filtering: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()