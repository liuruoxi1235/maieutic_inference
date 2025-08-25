import json
import re
import sys
import os
from pathlib import Path

def check_question_validity(question_text):
    """
    Check if a question text should be excluded based on formatting issues.
    
    Exclusion criteria:
    1. Contains single character followed by "." (like " a.", " b.", " 1.")
    2. Contains "..." (ellipsis)
    
    Returns (is_valid, reason)
    """
    if not question_text:
        return False, "Empty question text"
    
    # Check for "..." (ellipsis)
    if "..." in question_text:
        return False, "Contains ellipsis (...)"
    
    # Check for single character followed by "." 
    # Pattern: space + single character + period
    # This matches " a.", " b.", " 1.", " A.", etc.
    # But NOT " cry.", " the.", " etc." (multi-character words)
    single_char_pattern = r'\s[a-zA-Z0-9]\.'
    
    if re.search(single_char_pattern, question_text):
        # Find the actual match for reporting
        match = re.search(single_char_pattern, question_text)
        matched_text = match.group(0).strip() if match else "unknown"
        return False, f"Contains single character + period pattern: '{matched_text}'"
    
    return True, "Valid question format"

def load_mappings(mapping_file):
    """
    Load the variable mappings from JSON file
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

def filter_by_question_quality(mapping_file, output_file):
    """
    Filter variables based on question text quality.
    
    Excludes variables with:
    1. Single character + period patterns (like " a.", " b.", " 1.")
    2. Ellipsis ("...") in questions
    """
    print("GSS Question Text Quality Filter")
    print("=" * 50)
    print("Filtering criteria:")
    print("  ✗ Exclude questions with single character + period (e.g., ' a.', ' b.', ' 1.')")
    print("  ✗ Exclude questions with ellipsis (...)")
    print("  ✓ Keep questions with multi-character words + period (e.g., ' cry.', ' etc.')")
    print()
    
    # Load mappings
    mappings = load_mappings(mapping_file)
    
    # Filter variables based on question quality
    print(f"Filtering {len(mappings)} variables by question text quality...")
    print("-" * 60)
    
    filtered_mappings = {}
    stats = {
        'total_checked': 0,
        'valid_questions': 0,
        'has_ellipsis': 0,
        'has_single_char_period': 0,
        'empty_question': 0
    }
    
    # Track examples for reporting
    ellipsis_examples = []
    single_char_examples = []
    
    for var_name, var_info in mappings.items():
        stats['total_checked'] += 1
        
        question_text = var_info.get('question', '')
        
        # Check question validity
        is_valid, reason = check_question_validity(question_text)
        
        if is_valid:
            filtered_mappings[var_name] = var_info
            stats['valid_questions'] += 1
            status = "✓ INCLUDED"
        else:
            if "ellipsis" in reason.lower():
                stats['has_ellipsis'] += 1
                if len(ellipsis_examples) < 3:
                    ellipsis_examples.append((var_name, question_text[:60] + "..."))
            elif "single character" in reason.lower():
                stats['has_single_char_period'] += 1
                if len(single_char_examples) < 3:
                    single_char_examples.append((var_name, question_text[:60] + "..."))
            elif "empty" in reason.lower():
                stats['empty_question'] += 1
            
            status = "✗ EXCLUDED"
        
        # Print progress for key variables and every 20th variable
        if (stats['total_checked'] % 20 == 0 or 
            var_name.lower() in ['age', 'sex', 'race', 'educ', 'income', 'marital', 'class', 'polviews']):
            question_preview = question_text[:40] + "..." if len(question_text) > 40 else question_text
            print(f"{var_name:12} | {status:12} | {question_preview}")
            if not is_valid:
                print(f"{'':12} | {'':12} | Reason: {reason}")
    
    # Save filtered mappings
    print(f"\nSaving filtered mappings to: {output_file}")
    
    # Determine output format based on extension
    output_path = Path(output_file)
    
    if output_path.suffix.lower() == '.json':
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(filtered_mappings, f, indent=2, ensure_ascii=False)
    else:
        # Save as JSON with .json extension
        json_file = output_path.with_suffix('.json')
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(filtered_mappings, f, indent=2, ensure_ascii=False)
        output_file = str(json_file)
    
    print(f"✓ Saved {len(filtered_mappings)} variables with valid question text")
    
    # Print summary statistics
    print(f"\n=== QUESTION QUALITY FILTERING SUMMARY ===")
    print(f"Total variables checked: {stats['total_checked']}")
    print(f"Variables with valid questions: {stats['valid_questions']}")
    print(f"Excluded - Contains ellipsis (...): {stats['has_ellipsis']}")
    print(f"Excluded - Single char + period: {stats['has_single_char_period']}")
    print(f"Excluded - Empty question: {stats['empty_question']}")
    print(f"Retention rate: {stats['valid_questions']/stats['total_checked']*100:.1f}%")
    
    # Show examples of excluded patterns
    if ellipsis_examples:
        print(f"\n=== ELLIPSIS EXAMPLES (EXCLUDED) ===")
        for var_name, question in ellipsis_examples:
            print(f"{var_name:12} | {question}")
    
    if single_char_examples:
        print(f"\n=== SINGLE CHAR + PERIOD EXAMPLES (EXCLUDED) ===")
        for var_name, question in single_char_examples:
            print(f"{var_name:12} | {question}")
    
    # Show examples of kept variables
    print(f"\n=== SAMPLE VARIABLES WITH VALID QUESTIONS ===")
    count = 0
    for var_name, var_info in filtered_mappings.items():
        if count < 5:
            question = var_info.get('question', 'No description')
            data_type = var_info.get('data_type', 'unknown')
            domain_count = len(var_info.get('domain_values', []))
            
            # Show first 50 chars of question
            question_preview = question[:50] + "..." if len(question) > 50 else question
            print(f"{var_name:12} | {data_type:8} | {domain_count:2}d | {question_preview}")
            count += 1
        else:
            break
    
    if len(filtered_mappings) > 5:
        print(f"... and {len(filtered_mappings) - 5} more variables with valid questions")
    
    return len(filtered_mappings)

def test_question_patterns():
    """
    Test the question validation function with various examples
    """
    print("Testing question validation patterns...")
    print("=" * 40)
    
    test_cases = [
        # Should be EXCLUDED (single char + period)
        ("Question with a. option", False),
        ("Select b. from the list", False),
        ("Choose 1. or 2.", False),
        ("Rate A. through F.", False),
        ("Item z. is correct", False),
        
        # Should be EXCLUDED (ellipsis)
        ("This is a question...", False),
        ("Tell me about... your feelings", False),
        ("Rate from 1 to 5...", False),
        
        # Should be INCLUDED (valid questions)
        ("This person should cry.", True),
        ("Rate the etc. factor", True),
        ("How often do you try.", True),
        ("Select the appropriate option.", True),
        ("What is your age?", True),
        ("Do you agree or disagree?", True),
        ("Rate from 1 to 5 where 1 means strongly disagree", True),
        
        # Edge cases
        ("", False),  # Empty
        ("Just text without periods", True),
        ("Text with U.S. abbreviation", True),  # Multi-char before period
        ("Text with Dr. title", True),  # Multi-char before period
    ]
    
    print("Test case results:")
    for question, expected_valid in test_cases:
        is_valid, reason = check_question_validity(question)
        status = "✓" if is_valid == expected_valid else "✗ FAILED"
        print(f"{status} '{question[:30]}...' -> Valid: {is_valid} ({reason})")
    
    print("\nPattern matching examples:")
    single_char_pattern = r'\s[a-zA-Z0-9]\.'
    test_strings = [
        "Question with a.",  # Should match
        "Question with cry.",  # Should NOT match
        "Select b. option",  # Should match
        "Use etc. here",  # Should NOT match
        "Item 1. is first",  # Should match
        "Year 2023. was good"  # Should NOT match (multi-digit)
    ]
    
    for test_str in test_strings:
        match = re.search(single_char_pattern, test_str)
        result = f"MATCH: '{match.group(0).strip()}'" if match else "NO MATCH"
        print(f"'{test_str}' -> {result}")

def main():
    """
    Main function with command line interface
    """
    if len(sys.argv) == 2 and sys.argv[1] == "--test":
        test_question_patterns()
        return
    
    if len(sys.argv) != 3:
        print("Usage: python gss_question_filter.py <mapping_file> <output_file>")
        print("       python gss_question_filter.py --test")
        print()
        print("Arguments:")
        print("  mapping_file  : Path to the GSS mappings JSON file")
        print("  output_file   : Path for the filtered output file (.json)")
        print()
        print("Filtering criteria:")
        print("  - Excludes questions with single character + period (e.g., ' a.', ' b.', ' 1.')")
        print("  - Excludes questions with ellipsis (...)")
        print("  - Keeps questions with multi-character words + period (e.g., ' cry.', ' etc.')")
        print()
        print("Examples:")
        print("  python gss_question_filter.py gss2022_filtered.json gss2022_clean_questions.json")
        print("  python gss_question_filter.py --test")
        sys.exit(1)
    
    mapping_file = sys.argv[1]
    output_file = sys.argv[2]
    
    # Validate input file exists
    if not os.path.exists(mapping_file):
        print(f"❌ Mapping file not found: {mapping_file}")
        sys.exit(1)
    
    # Run the filtering
    try:
        num_filtered = filter_by_question_quality(mapping_file, output_file)
        print(f"\n🎉 Successfully filtered to {num_filtered} variables with clean question text!")
        print("Variables excluded had formatting issues (single char + period or ellipsis)")
        
    except KeyboardInterrupt:
        print("\n⚠️ Process interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error during filtering: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()