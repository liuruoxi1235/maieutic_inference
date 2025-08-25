import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
import time
from urllib.parse import urljoin
import os
import warnings
from collections import defaultdict

# Suppress SSL warnings when using verify=False
warnings.filterwarnings('ignore', message='Unverified HTTPS request')

def download_page(url, output_file=None):
    """
    Downloads a page and optionally saves it to a file
    
    Args:
        url: URL to download
        output_file: Optional path to save the HTML
        
    Returns:
        BeautifulSoup object for the page
    """
    try:
        print(f"Downloading {url}")
        response = requests.get(url)
        
        if response.status_code == 200:
            if output_file:
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(response.text)
            
            return BeautifulSoup(response.text, 'html.parser')
        else:
            print(f"Failed to download {url}: {response.status_code}")
            return None
    
    except Exception as e:
        print(f"Error downloading {url}: {str(e)}")
        return None

def clean_question_text(text):
    """
    Clean up question text by removing prefixes and extra whitespace
    
    Args:
        text: The raw question text
        
    Returns:
        Cleaned question text
    """
    # Check if there's a period in the first 8 characters (like "1603a.")
    if '.' in text[:8]:
        # Find the first period and remove everything before it and including it
        first_period_pos = text.find('.')
        text = text[first_period_pos+1:].strip()
    else:
        # Try the standard number + period pattern (like "13.")
        text = re.sub(r'^\d+\.\s+', '', text)
    
    # Replace any line breaks or multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def extract_variables_from_page(soup):
    """
    Extracts all variables and their questions/descriptions from a page
    
    Args:
        soup: BeautifulSoup object of the page
        
    Returns:
        Dictionary of variable_name: question/description
    """
    if not soup:
        return {}
    
    variable_info = {}
    
    # Find all variable blocks
    var_blocks = soup.find_all('div', {'class': 'vardesc'})
    print(f"Found {len(var_blocks)} variable blocks on page")
    
    for block in var_blocks:
        # Find the variable name
        varname_span = block.find('span', {'class': 'varname'})
        if not varname_span:
            continue
        
        variable_name = varname_span.get_text().strip()
        
        # First try to get the question text if it exists
        question_text = None
        
        # Look for a div containing the question text heading
        question_heading = None
        for span in block.find_all('span', {'class': 'genericheading'}):
            if "Text of this Question or Item" in span.get_text():
                question_heading = span
                break
        
        if question_heading:
            # Find the parent div and the pre tag with the question
            question_div = question_heading.parent
            if question_div and question_div.find('pre'):
                pre_text = question_div.find('pre').get_text().strip()
                # Clean up the question using our helper function
                question_text = clean_question_text(pre_text)
        
        # If no question text found, use the variable label instead
        if not question_text:
            varlabel_span = block.find('span', {'class': 'varlabel'})
            if varlabel_span:
                question_text = varlabel_span.get_text().strip()
        
        # Store the variable and question/description
        if variable_name and question_text:
            variable_info[variable_name] = question_text
            print(f"✓ Found for {variable_name}: {question_text[:50]}...")
    
    return variable_info

def remove_duplicate_descriptions(variable_info):
    """
    Remove variables with duplicate descriptions
    
    Args:
        variable_info: Dictionary of variable_name: question/description
        
    Returns:
        Dictionary with duplicate descriptions removed
    """
    # Group variables by description
    desc_to_vars = defaultdict(list)
    for var, desc in variable_info.items():
        desc_to_vars[desc].append(var)
    
    # Create a new dictionary without variables that have duplicate descriptions
    unique_variable_info = {}
    duplicates_removed = []
    
    for desc, vars_list in desc_to_vars.items():
        if len(vars_list) == 1:
            # Description is unique, keep the variable
            unique_variable_info[vars_list[0]] = desc
        else:
            # Description appears multiple times, remove all matching variables
            duplicates_removed.extend(vars_list)
    
    if duplicates_removed:
        print(f"\nRemoved {len(duplicates_removed)} variables with duplicate descriptions:")
        for var in sorted(duplicates_removed):
            print(f"  - {var}")
    
    return unique_variable_info

def scrape_pages(output_file, page_limit=None, html_file=None):
    """
    Scrapes variable information from GSS codebook pages
    
    Args:
        output_file: Path to save results
        page_limit: Maximum number of pages to scrape (None = no limit)
        html_file: Optional local HTML file to process
    """
    # Create directory for HTML files
    os.makedirs("html_files", exist_ok=True)
    
    # Dictionary to store all variable info
    all_variables = {}
    
    # If HTML file provided, process it first
    if html_file:
        print(f"Processing local HTML file: {html_file}")
        with open(html_file, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        local_soup = BeautifulSoup(html_content, 'html.parser')
        local_variables = extract_variables_from_page(local_soup)
        all_variables.update(local_variables)
        
        # Save and return if we're only processing a local file
        save_results(all_variables, output_file)
        return all_variables
    
    # Process online pages
    base_url = "https://sda.berkeley.edu/D3/GSS18/Doc/"
    total_pages = 0  # Count total pages processed
    
    if page_limit is None:
        # Scrape all hcbk pages
        print("No limit specified. Attempting to scrape all 'hcbk' pages...")
        
        # Try sequential numbering first (hcbk0001.htm, hcbk0002.htm, etc.)
        page_num = 1
        sequential_failed = False
        
        while not sequential_failed:
            url = urljoin(base_url, f"hcbk{page_num:04d}.htm")
            print(f"Trying {url}")
            
            page_soup = download_page(url, output_file=f"html_files/hcbk{page_num:04d}.htm")
            
            if page_soup:
                total_pages += 1
                page_variables = extract_variables_from_page(page_soup)
                if page_variables:
                    all_variables.update(page_variables)
                page_num += 1
                time.sleep(0.5)
            else:
                sequential_failed = True
                print(f"Sequential page {page_num} not found, trying alternate patterns...")
        
        # Try alphabetical pages (hcbkA.htm through hcbkZ.htm)
        for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
            url = urljoin(base_url, f"hcbk{letter}.htm")
            print(f"Trying {url}")
            
            page_soup = download_page(url, output_file=f"html_files/hcbk{letter}.htm")
            
            if page_soup:
                total_pages += 1
                page_variables = extract_variables_from_page(page_soup)
                if page_variables:
                    all_variables.update(page_variables)
                time.sleep(0.5)
        
        # Try function pages (hcbkfx0.htm, etc.) and index pages (hcbkx01.htm, etc.)
        for prefix in ['fx', 'x']:
            for i in range(10):  # Try a few numbers
                url = urljoin(base_url, f"hcbk{prefix}{i}.htm")
                print(f"Trying {url}")
                
                page_soup = download_page(url, output_file=f"html_files/hcbk{prefix}{i}.htm")
                
                if page_soup:
                    total_pages += 1
                    page_variables = extract_variables_from_page(page_soup)
                    if page_variables:
                        all_variables.update(page_variables)
                    time.sleep(0.5)
    else:
        # We have a page limit, use the "Next Page" approach
        current_url = urljoin(base_url, "hcbk0001.htm")
        
        while total_pages < page_limit:
            print(f"[{total_pages+1}/{page_limit}] Processing {current_url}")
            
            # Download and process the page
            page_soup = download_page(current_url, output_file=f"html_files/page_{total_pages+1}.htm")
            
            if not page_soup:
                print(f"Failed to process {current_url}, stopping.")
                break
            
            # Extract variables from this page
            page_variables = extract_variables_from_page(page_soup)
            if page_variables:
                all_variables.update(page_variables)
            
            # Count this page
            total_pages += 1
            
            # Check if we've reached the limit
            if total_pages >= page_limit:
                break
            
            # Find the next page link
            next_page = None
            try:
                next_page_link = page_soup.find('a', string=re.compile('Next Page', re.IGNORECASE))
                if next_page_link:
                    next_page_href = next_page_link.get('href')
                    if next_page_href:
                        current_url = urljoin(base_url, next_page_href)
                        print(f"Found next page link: {current_url}")
                    else:
                        print("No href attribute in next page link, stopping.")
                        break
                else:
                    print("No next page link found, stopping.")
                    break
            except Exception as e:
                print(f"Error finding next page: {str(e)}")
                break
            
            # Add a small delay
            time.sleep(0.5)
    
    print(f"\nTotal pages processed: {total_pages}")
    print(f"Total variables found: {len(all_variables)}")
    
    # Remove variables with duplicate descriptions
    print("\nChecking for duplicate descriptions...")
    unique_variables = remove_duplicate_descriptions(all_variables)
    print(f"Variables after removing duplicates: {len(unique_variables)}")
    
    # Save results
    save_results(unique_variables, output_file)
    
    return unique_variables

def save_results(variable_info, output_file):
    """
    Save the variable information to files
    
    Args:
        variable_info: Dictionary of variable_name: question/description
        output_file: Base name for output files
    """
    # Create DataFrame
    df = pd.DataFrame([
        {'Variable': var, 'Question': clean_question_text(question)}
        for var, question in variable_info.items()
    ])
    
    # Sort by variable name
    df = df.sort_values('Variable')
    
    # Save to CSV
    df.to_csv(f"{output_file}.csv", index=False)
    
    # Clean all questions and save to text file
    with open(output_file, 'w', encoding='utf-8') as f:
        for var, question in sorted(variable_info.items()):
            cleaned_question = clean_question_text(question)
            f.write(f"{var}: {cleaned_question}\n")
    
    print(f"Results saved to {output_file} and {output_file}.csv")

def process_local_file(html_file, output_file):
    """
    Process a local HTML file to extract variable-question pairs
    
    Args:
        html_file: Path to the HTML file
        output_file: Path to the output file
    """
    # Process the local file
    scrape_pages(output_file, html_file=html_file)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Scrape GSS variable-question pairs")
    parser.add_argument("--output", "-o", default="gss_variables.txt", help="Output file path")
    parser.add_argument("--limit", "-l", type=int, help="Maximum number of pages to scrape (optional)")
    parser.add_argument("--html-file", "-f", help="Process a local HTML file instead of scraping")
    
    args = parser.parse_args()
    
    if args.html_file:
        process_local_file(args.html_file, args.output)
    else:
        scrape_pages(args.output, args.limit)