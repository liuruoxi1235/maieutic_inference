import requests
import re
from bs4 import BeautifulSoup
import json
import time
import pandas as pd

def scrape_all_gss_2024_with_domains():
    """
    Extract variables from GSS 2024 with domains - DEBUG VERSION (First 5 pages only)
    Fixed to properly extract questions from <pre> tags and domains from table headers
    """
    
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    })
    
    base_url = "https://sda.berkeley.edu/sdaweb/docs/gss24rel1/DOC/"
    all_variables = {}
    
    print("Scraping GSS 2024 variables with domains - DEBUG VERSION")
    print("DEBUGGING: Limited to first 5 pages only")
    print("Fixed parsing for questions and domain values\n")
    
    # DEBUG: Limit to first 5 pages
    max_pages = 1000
    page_num = 1
    consecutive_404s = 0
    
    while page_num <= max_pages and consecutive_404s < 3:
        page_file = f"hcbk{page_num:04d}.htm"
        page_url = f"{base_url}{page_file}"
        
        try:
            print(f"DEBUG: Processing page {page_num}/{max_pages}: {page_file}")
            response = session.get(page_url, timeout=15)
            
            if response.status_code == 404:
                consecutive_404s += 1
                print(f"  404 Not Found ({consecutive_404s}/3)")
                page_num += 1
                continue
            else:
                consecutive_404s = 0
            
            if response.status_code == 200:
                page_variables = extract_variables_from_html(response.text, page_num)
                
                for var_name, var_info in page_variables.items():
                    all_variables[var_name] = var_info
                
                print(f"  Found {len(page_variables)} variables")
                
                # Show samples with improved parsing
                if page_variables:
                    for i, (var_name, info) in enumerate(page_variables.items()):
                        if i < 3:
                            question = info.get('question', 'No question')[:80]
                            domain = info.get('domain_values', [])
                            domain_str = ', '.join(domain[:4]) if domain else 'No domain'
                            print(f"  {var_name}: {question}...")
                            print(f"    Domain: {domain_str}")
                        else:
                            break
            
            page_num += 1
            time.sleep(0.3)  # Be gentle on the server
            
        except Exception as e:
            print(f"Error on page {page_num}: {e}")
            consecutive_404s += 1
            page_num += 1
            continue
    
    print(f"\nDEBUG: Finished scraping {min(page_num-1, max_pages)} pages")
    print(f"Total variables found: {len(all_variables)}")
    return all_variables

def extract_variables_from_html(html_content, page_num):
    """
    Extract variables from HTML using corrected parsing logic
    """
    soup = BeautifulSoup(html_content, 'html.parser')
    variables = {}
    
    # Extract current topic/domain context from headers
    current_topic = extract_topic_context(soup, page_num)
    
    # Find all variable sections (div with class="vardesc")
    var_sections = soup.find_all('div', class_='vardesc')
    
    print(f"    Found {len(var_sections)} variable sections")
    
    for section in var_sections:
        var_info = parse_variable_section_fixed(section, current_topic)
        if var_info:
            var_name = var_info['variable_name']
            variables[var_name] = var_info
    
    return variables

def extract_topic_context(soup, page_num):
    """
    Extract topic/domain context from page headers (h1, h2, h3)
    Note: This is now only used for debugging output, not stored in final data
    """
    # Look for meaningful headers that indicate topic areas
    headers = soup.find_all(['h1', 'h2', 'h3'])
    
    topic_parts = []
    
    for header in headers:
        text = header.get_text().strip()
        
        # Skip navigation and generic headers
        if any(skip in text.lower() for skip in [
            'gss', 'cumulative', 'datafile', 'release', 'title page', 'indexes'
        ]):
            continue
        
        # Include meaningful topic headers
        if len(text) > 5 and not text.isupper():
            topic_parts.append(text)
        elif text.isupper() and len(text) > 10:
            # Handle ALL CAPS headers like "RESPONDENT BACKGROUND VARIABLES"
            topic_parts.append(text.title())
    
    # Combine topic parts or use fallback (only for debug display)
    if topic_parts:
        return " - ".join(topic_parts[-2:])  # Use last 2 levels
    else:
        return f"Page {page_num} Variables"

def parse_variable_section_fixed(section, default_topic):
    """
    Parse a single variable section with corrected logic
    """
    # Extract variable name
    var_name_span = section.find('span', class_='varname')
    if not var_name_span:
        return None
    
    var_name = var_name_span.get_text().strip()
    
    # Extract variable label
    var_label_span = section.find('span', class_='varlabel')
    var_label = var_label_span.get_text().strip() if var_label_span else ""
    
    # Extract the actual question from <pre> tag
    question_text = extract_question_fixed(section)
    
    # If no question in <pre>, fall back to variable label
    if not question_text and var_label:
        question_text = var_label
    
    # Extract domain values from table headers
    domain_values = extract_domain_values_fixed(section)
    
    # DEBUG: Print what we found for key variables
    if var_name in ['sex', 'race', 'age']:
        print(f"    DEBUG {var_name}: Found {len(domain_values)} domain values: {domain_values[:5]}")
    
    # Extract additional metadata
    metadata = extract_variable_metadata(section)
    
    # Determine data type: discrete if domain_values exist, otherwise use metadata or default to numeric
    if domain_values:
        data_type = "discrete"
    else:
        data_type = metadata.get('data_type', 'numeric')
    
    # CLEANED: Organize data structure without redundant data_type in metadata
    # Move range from metadata to main level
    result = {
        'variable_name': var_name,
        'question': question_text or 'No description available',
        'variable_label': var_label,
        'domain_values': domain_values,
        'data_type': data_type
    }
    
    # Add range if available
    if metadata.get('range'):
        result['range'] = metadata['range']
    
    # Add remaining metadata (excluding data_type and range which are now at main level)
    remaining_metadata = {k: v for k, v in metadata.items() if k not in ['data_type', 'range']}
    if remaining_metadata:
        result['metadata'] = remaining_metadata
    
    return result

def extract_question_fixed(section):
    """
    Extract the survey question from <pre> tag after "Text of this Question or Item"
    """
    # Look for the exact text indicator
    text_elements = section.find_all(text=re.compile(r'Text of this Question or Item', re.IGNORECASE))
    
    for text_element in text_elements:
        # Find the parent and look for the next <pre> tag
        parent = text_element.parent
        if parent:
            # Look for <pre> tag that comes after this indicator
            pre_tag = parent.find_next_sibling('pre')
            if not pre_tag:
                # Sometimes the <pre> is nested, so search more broadly
                next_elements = parent.find_next_siblings()
                for elem in next_elements[:3]:  # Check next few elements
                    pre_tag = elem.find('pre') if hasattr(elem, 'find') else None
                    if pre_tag:
                        break
            
            if pre_tag:
                question_text = pre_tag.get_text().strip()
                # Clean up the question text
                question_text = re.sub(r'^\d+[a-z]?\.\s*', '', question_text)  # Remove question numbers
                question_text = re.sub(r'\s+', ' ', question_text)  # Normalize whitespace
                question_text = question_text.replace('\n', ' ').strip()
                return question_text
    
    # Alternative: look for any <pre> tag in the section that looks like a question
    pre_tags = section.find_all('pre')
    for pre_tag in pre_tags:
        text = pre_tag.get_text().strip()
        if len(text) > 10:  # Substantial content
            text = re.sub(r'^\d+[a-z]?\.\s*', '', text)
            text = re.sub(r'\s+', ' ', text)
            text = text.replace('\n', ' ').strip()
            return text
    
    return None

def extract_domain_values_fixed(section):
    """
    Extract domain values from table headers like <td style="width: 50px;">1<br>male</td>
    This should extract: [male, female, don't know, iap, no answer, skipped on web]
    """
    domain_values = []
    
    # Find tables with class="dflt" - these contain the domain headers
    tables = section.find_all('table', class_='dflt')
    
    if not tables:
        # Fallback to any table if no dflt class found
        tables = section.find_all('table')
    
    for table in tables:
        rows = table.find_all('tr')
        
        # Look for the row with domain headers - usually contains cells with "width: 50px"
        for row_idx, row in enumerate(rows):
            cells = row.find_all(['td', 'th'])
            domain_cells_found = []
            
            for cell in cells:
                cell_style = str(cell.get('style', ''))
                
                # Look for cells with "width: 50px" style - these are domain value headers
                if 'width: 50px' in cell_style:
                    cell_html = str(cell)
                    
                    # Debug: print the cell HTML for troubleshooting
                    if len(domain_cells_found) < 3:  # Only print first few
                        print(f"      DEBUG cell HTML: {cell_html[:100]}...")
                    
                    # Parse the pattern: <td style="width: 50px;">1<br>male</td>
                    br_match = re.search(r'>([^<]*)<br[^>]*>([^<]*)<', cell_html)
                    if br_match:
                        code = br_match.group(1).strip()
                        label = br_match.group(2).strip()
                        
                        # Clean up HTML entities
                        label = label.replace('&#x27;', "'").replace('&amp;', '&')
                        
                        # Skip empty labels or row headers
                        if label and label.lower() not in ['row', 'total', 'cells contain', '']:
                            domain_cells_found.append(label)
                    else:
                        # Try alternative parsing for simpler cases
                        cell_text = cell.get_text(separator='|').strip()
                        if '|' in cell_text:
                            parts = cell_text.split('|')
                            if len(parts) >= 2 and parts[1].strip():
                                label = parts[1].strip()
                                if label.lower() not in ['row', 'total', 'cells contain', '']:
                                    domain_cells_found.append(label)
            
            # If we found domain cells in this row, use them and stop
            if domain_cells_found:
                domain_values.extend(domain_cells_found)
                print(f"      DEBUG: Found {len(domain_cells_found)} domain values in row {row_idx}")
                break
    
    # Remove duplicates while preserving order
    seen = set()
    unique_values = []
    for value in domain_values:
        if value not in seen:
            seen.add(value)
            unique_values.append(value)
    
    return unique_values

def extract_variable_metadata(section):
    """
    Extract additional metadata like data type, range, etc.
    """
    metadata = {}
    
    # Look for data type
    data_type_text = section.find(text=re.compile(r'Data type:', re.IGNORECASE))
    if data_type_text:
        parent = data_type_text.parent
        if parent:
            next_cell = parent.find_next_sibling(['td', 'th'])
            if next_cell:
                metadata['data_type'] = next_cell.get_text().strip()
    
    # Look for range information
    range_text = section.find(text=re.compile(r'Range of valid codes:', re.IGNORECASE))
    if range_text:
        # Extract the range from the parent text
        parent_text = range_text.parent.get_text() if range_text.parent else ""
        range_match = re.search(r'Range of valid codes:\s*([^)]+)', parent_text)
        if range_match:
            metadata['range'] = range_match.group(1).strip()
    
    # Look for total cases
    total_text = section.find(text=re.compile(r'Total Cases:', re.IGNORECASE))
    if total_text:
        parent_text = total_text.parent.get_text() if total_text.parent else ""
        total_match = re.search(r'(\d{1,3}(?:,\d{3})*)', parent_text)
        if total_match:
            metadata['total_cases'] = total_match.group(1)
    
    return metadata

def save_results_with_domains(variables, filename_base="gss2024_debug_5pages_fixed"):
    """Save variables with corrected parsing - DEBUG VERSION"""
    
    if not variables:
        print("No variables found to save.")
        return
    
    # Prepare data for different formats
    simple_mapping = {}  # var_name -> question
    domain_mapping = {}  # var_name -> domain values
    detailed_data = []   # Full information for CSV
    
    for var_name, info in variables.items():
        question = info.get('question', 'No description')
        domain_values = info.get('domain_values', [])
        data_type = info.get('data_type', 'numeric')
        metadata = info.get('metadata', {})
        range_info = info.get('range', '')
        
        # Format domain values for display
        domain_display = ', '.join(domain_values[:8]) if domain_values else 'No domain values'
        
        simple_mapping[var_name] = question
        domain_mapping[var_name] = domain_display
        
        detailed_data.append({
            'variable_name': var_name,
            'question_text': question,
            'data_type': data_type,
            'range': range_info,
            'domain_values': '; '.join(domain_values),
            'variable_label': info.get('variable_label', ''),
            'total_cases': metadata.get('total_cases', ''),
            'num_domain_values': len(domain_values)
        })
    
    # Save outputs
    json_simple = f"{filename_base}_questions.json"
    with open(json_simple, 'w', encoding='utf-8') as f:
        json.dump(simple_mapping, f, indent=2, ensure_ascii=False)
    print(f"✓ Saved variable->question mapping to {json_simple}")
    
    json_domains = f"{filename_base}_domains.json"
    with open(json_domains, 'w', encoding='utf-8') as f:
        json.dump(domain_mapping, f, indent=2, ensure_ascii=False)
    print(f"✓ Saved variable->domain mapping to {json_domains}")
    
    # Save complete data with cleaned structure
    json_complete = f"{filename_base}_complete.json"
    complete_mapping = {}
    for var_name, info in variables.items():
        # CLEANED: Use the new organized structure
        var_data = {
            'question': info.get('question', 'No description'),
            'domain_values': info.get('domain_values', []),
            'data_type': info.get('data_type', 'numeric'),
            'variable_label': info.get('variable_label', '')
        }
        
        # Add range if available
        if info.get('range'):
            var_data['range'] = info['range']
        
        # Add remaining metadata if any
        if info.get('metadata'):
            var_data['metadata'] = info['metadata']
            
        complete_mapping[var_name] = var_data
    
    with open(json_complete, 'w', encoding='utf-8') as f:
        json.dump(complete_mapping, f, indent=2, ensure_ascii=False)
    print(f"✓ Saved complete mapping to {json_complete}")
    
    # Save CSV
    csv_file = f"{filename_base}.csv"
    df = pd.DataFrame(detailed_data)
    df.to_csv(csv_file, index=False, encoding='utf-8')
    print(f"✓ Saved {len(variables)} variables to {csv_file}")
    
    # Print statistics
    print(f"\n=== DEBUG EXTRACTION STATISTICS (First 5 Pages - CLEANED) ===")
    print(f"Total variables extracted: {len(variables)}")
    
    # Count by data type
    type_counts = {}
    for info in variables.values():
        data_type = info.get('data_type', 'unknown')
        type_counts[data_type] = type_counts.get(data_type, 0) + 1
    
    print(f"\nVariables by data type:")
    for data_type, count in sorted(type_counts.items()):
        print(f"  {data_type}: {count}")
    
    # Show sample results with cleaned structure
    print(f"\n=== SAMPLE CLEANED VARIABLES ===")
    for i, (var_name, info) in enumerate(variables.items()):
        if i < 5:
            question = info.get('question', 'No description')
            data_type = info.get('data_type', 'unknown')
            domain_values = info.get('domain_values', [])
            range_info = info.get('range', '')
            
            print(f"\n{i+1}. Variable: {var_name}")
            print(f"   Data Type: {data_type}")
            print(f"   Question: {question}")
            print(f"   Domain: {', '.join(domain_values[:5])}{'...' if len(domain_values) > 5 else ''}")
            if range_info:
                print(f"   Range: {range_info}")
        else:
            break
    
    if len(variables) > 5:
        print(f"\n... and {len(variables) - 5} more variables")
    
    return len(variables)

if __name__ == "__main__":
    print("DEBUG VERSION: GSS 2024 Scraper with CLEANED Data Structure")
    print("=" * 60)
    print("🔍 DEBUG MODE: Only scraping first 5 pages for testing")
    print("🔧 CLEANED: Removed redundant data_type from metadata")
    print("🔧 CLEANED: Moved range from metadata to main level")
    print("Starting from page 1, limited to 5 pages\n")
    
    # Extract variables with cleaned structure - DEBUG VERSION
    all_variables = scrape_all_gss_2024_with_domains()
    
    if all_variables:
        num_saved = save_results_with_domains(all_variables)
        print(f"\n🎉 DEBUG: Successfully extracted {num_saved} variables with CLEANED structure!")
        print("\nCleaned debug files created:")
        print("  - gss2024_debug_5pages_fixed_questions.json (variable -> question)")
        print("  - gss2024_debug_5pages_fixed_domains.json (variable -> domain values)")
        print("  - gss2024_debug_5pages_fixed_complete.json (variable -> {question, domain_values, data_type, variable_label, range, metadata})")
        print("  - gss2024_debug_5pages_fixed.csv (complete spreadsheet)")
        print("\n🧹 CLEANED STRUCTURE:")
        print("  ✓ Removed redundant data_type from metadata")
        print("  ✓ Moved range from metadata to main level (parallel to variable_label, question, etc.)")
        print("  ✓ Metadata now only contains remaining fields (e.g., total_cases)")
        print("\n📋 Once debugging is complete, remove the page limit to scrape all pages.")
    else:
        print("\n❌ No variables extracted. Check connection or URL structure.")