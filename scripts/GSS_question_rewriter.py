import json
import sys
import os
import re
from pathlib import Path
from openai import OpenAI
import time

class GSSQuestionReviser:
    def __init__(self, api_key, model_id="gpt-4o", temperature=0.3, log_path="gss_revision.log"):
        """
        Initialize the GSS Question Reviser
        """
        self.client = OpenAI(api_key=api_key)
        self.model_ID = model_id
        self.temperature = temperature
        self.log_pth = log_path
        self.LLM_budget_left = 10000  # Adjust as needed
        
        # Clear log file
        with open(self.log_pth, "w") as log_file:
            log_file.write("GSS Question Revision Log\n" + "="*50 + "\n\n")
    
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

        self.LLM_budget_left -= 1

        # Return the actual generation
        return response.choices[0].message.content
    
    def get_revision_prompt(self):
        """
        Define the prompt template for question revision
        """
        return [
            ("system", """You are an expert survey methodologist tasked with revising GSS (General Social Survey) questions to make them more readable and understandable for general audiences.

Your job is to:
1. Take a GSS variable with its original question text (and domain for your reference)
2. Revise the question to a more concise, human-readable question WITHOUT changing the original meaning. Address the respondent as "the respondent" instead of "you", "he" or "r". 
3. Ensure the revised question can stand alone without survey context
4. Optinally, you can reject variables if you believe they are survey-administrative or context-dependent (for example, follow-up questions saying "If xxx is true, then ...")
The revised question do not twist the meaning of the original question, while at the same time the revised questions should not be too long. Also, the values in the domain should mean the same thing -- for example, question "Are you married" cannot be rewritten into "Are you devorced" because it changes what "Yes" and "No" means for this question.
             
REVISION GUIDELINES:
- Make questions clear and conversational
- Remove survey jargon and technical language
- Preserve the exact meaning and intent of the original
- Keep questions concise but complete
- Ensure questions make sense without additional context

REJECTION CRITERIA - Reject if the question:
- Is about the survey itself (timing, format, interviewer, etc.)
- Cannot stand alone without survey context
- Is purely administrative (ID numbers, technical codes)
- Refers to previous questions or survey flow
Note that Rejection is rare. Most questions are good and stand by itself.

OUTPUT FORMAT:
If ACCEPTING the variable, respond with exactly:
ACCEPT: [your revised question here]

If REJECTING the variable, respond with exactly:
REJECT: [brief reason for rejection]

Examples:
- "ACCEPT: How often does the respondent attend religious services?"
- "REJECT: Survey administrative question about interviewer"
"""),
            
            ("user", """Please revise this GSS variable or reject it if inappropriate:

Original Question: {question}
Domain Type (numeric or discrete): {data_type}
Domain Values (for discrete questions): {domain_values}
Range (for numeric questions): {range}

Please provide your response in the exact format specified.""")
        ]
    
    def parse_llm_response(self, response, var_name):
        """
        Parse the LLM response to extract decision and revised question
        """
        response = response.strip()
        
        if response.startswith("ACCEPT:"):
            revised_question = response[7:].strip()
            if revised_question:
                return "accept", revised_question
            else:
                with open(self.log_pth, "a") as log_file:
                    log_file.write(f"[warning] {var_name}: Empty revised question, treating as reject\n")
                return "reject", "Empty revised question"
        
        elif response.startswith("REJECT:"):
            reason = response[7:].strip()
            return "reject", reason if reason else "No reason provided"
        
        else:
            # Try to extract from malformed response
            if "accept" in response.lower():
                # Try to find the question after "accept"
                match = re.search(r'accept:?\s*(.+)', response, re.IGNORECASE | re.DOTALL)
                if match:
                    revised_question = match.group(1).strip()
                    return "accept", revised_question
            
            if "reject" in response.lower():
                # Try to find the reason after "reject"
                match = re.search(r'reject:?\s*(.+)', response, re.IGNORECASE | re.DOTALL)
                if match:
                    reason = match.group(1).strip()
                    return "reject", reason
            
            # Default to reject for malformed responses
            with open(self.log_pth, "a") as log_file:
                log_file.write(f"[warning] {var_name}: Malformed response, treating as reject: {response}\n")
            return "reject", f"Malformed response: {response[:50]}..."
    
    def revise_single_variable(self, var_name, var_info):
        """
        Revise a single variable's question using ChatGPT
        """
        # Prepare replacements for the prompt
        replacements = {
            "question": var_info.get('question', 'No description'),
            "data_type": var_info.get('data_type', 'unknown'),
            "domain_values": ', '.join(var_info.get('domain_values', [])) if var_info.get('domain_values') else 'None',
            "range": var_info.get('range', 'Not specified')
        }
        
        # Get the prompt template
        prompt_template = self.get_revision_prompt()
        
        try:
            # Call ChatGPT
            response = self.single_LLM_call(
                prompt_template, 
                self.client, 
                f"revise_{var_name}", 
                replacements
            )
            
            # Parse the response
            decision, result = self.parse_llm_response(response, var_name)
            
            return decision, result
            
        except Exception as e:
            with open(self.log_pth, "a") as log_file:
                log_file.write(f"[error] {var_name}: API call failed: {str(e)}\n")
            return "error", f"API call failed: {str(e)}"
    
    def revise_all_questions(self, input_file, output_file, start_from=None, max_variables=100):
        """
        Revise questions for all variables in the mapping file
        """
        print("GSS Question Revision with ChatGPT 4o")
        print("=" * 50)
        print(f"Model: {self.model_ID}")
        print(f"Temperature: {self.temperature}")
        print(f"Log file: {self.log_pth}")
        print(f"Max variables to process: {max_variables}")
        print()
        
        # Load the mappings
        print(f"Loading mappings from: {input_file}")
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                mappings = json.load(f)
            print(f"✓ Loaded {len(mappings)} variables")
        except Exception as e:
            print(f"❌ Error loading mappings: {e}")
            return 0
        
        # Determine which variables to process
        var_items = list(mappings.items())
        
        if start_from:
            start_idx = 0
            for i, (var_name, _) in enumerate(var_items):
                if var_name == start_from:
                    start_idx = i
                    break
            var_items = var_items[start_idx:]
            print(f"Starting from variable: {start_from}")
        
        # Filter out variables that already have revision_status and limit to first 100
        original_count = len(var_items)
        var_items_to_process = []
        
        for var_name, var_info in var_items:
            if 'revision_status' in var_info:
                print(f"Skipping {var_name} (already has revision_status: {var_info['revision_status']})")
                continue
            var_items_to_process.append((var_name, var_info))
            if len(var_items_to_process) >= max_variables:
                break
        
        print(f"Found {original_count} total variables")
        print(f"Skipped {original_count - len(var_items_to_process)} variables (already processed)")
        print(f"Processing {len(var_items_to_process)} variables...")
        print("-" * 50)
        
        # Track statistics
        stats = {
            'total_processed': 0,
            'accepted': 0,
            'rejected': 0,
            'errors': 0,
            'skipped': original_count - len(var_items_to_process)
        }
        
        # Process each variable
        for i, (var_name, var_info) in enumerate(var_items_to_process):
            stats['total_processed'] += 1
            
            print(f"[{i+1:3d}/{len(var_items_to_process)}] Processing {var_name}...")
            
            # Call ChatGPT to revise the question
            decision, result = self.revise_single_variable(var_name, var_info)
            
            # Update the variable info based on decision
            if decision == "accept":
                var_info['revised_question'] = result
                var_info['revision_status'] = 'accepted'
                stats['accepted'] += 1
                print(f"    ✓ ACCEPTED: {result[:60]}{'...' if len(result) > 60 else ''}")
                
            elif decision == "reject":
                var_info['revised_question'] = None
                var_info['revision_status'] = 'rejected'
                var_info['rejection_reason'] = result
                stats['rejected'] += 1
                print(f"    ✗ REJECTED: {result}")
                
            else:  # error
                var_info['revised_question'] = None
                var_info['revision_status'] = 'error'
                var_info['error_message'] = result
                stats['errors'] += 1
                print(f"    ❌ ERROR: {result}")
            
            # Update the mappings dictionary
            mappings[var_name] = var_info
            
            # Save progress every 10 variables
            if stats['total_processed'] % 10 == 0:
                self.save_progress(mappings, output_file, stats)
            
            # Rate limiting: small delay between API calls
            time.sleep(0.5)
            
            # Check budget
            if self.LLM_budget_left <= 0:
                print("\n⚠️ LLM budget exhausted, stopping processing")
                break
        
        # Final save
        self.save_progress(mappings, output_file, stats, final=True)
        
        return stats
    
    def save_progress(self, mappings, output_file, stats, final=False):
        """
        Save current progress to the output file
        """
        try:
            output_path = Path(output_file)
            if output_path.suffix.lower() != '.json':
                output_path = output_path.with_suffix('.json')
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(mappings, f, indent=2, ensure_ascii=False)
            
            status = "Final save" if final else "Progress save"
            print(f"    {status}: {output_path}")
            
        except Exception as e:
            print(f"    ❌ Save error: {e}")
            with open(self.log_pth, "a") as log_file:
                log_file.write(f"[error] Save failed: {str(e)}\n")

def load_api_key():
    """
    Load OpenAI API key from environment variable or file
    """
    # Try environment variable first
    api_key = os.getenv('OPENAI_API_KEY')
    if api_key:
        return api_key
    
    # Try loading from file
    api_key_files = ['openai_api_key.txt', 'api_key.txt', '.env']
    for filename in api_key_files:
        if os.path.exists(filename):
            try:
                with open(filename, 'r') as f:
                    content = f.read().strip()
                    # Handle .env format
                    if '=' in content:
                        for line in content.split('\n'):
                            if line.startswith('OPENAI_API_KEY='):
                                return line.split('=', 1)[1].strip().strip('"\'')
                    else:
                        return content
            except Exception:
                continue
    
    return None

def main():
    """
    Main function with command line interface
    """
    if len(sys.argv) < 3:
        print("Usage: python gss_question_reviser.py <input_mapping_file> <output_file> [options]")
        print()
        print("Arguments:")
        print("  input_mapping_file : Path to the GSS mappings JSON file")
        print("  output_file        : Path for the output file with revised questions")
        print()
        print("Options:")
        print("  --start-from VAR   : Start processing from specific variable name")
        print("  --max-vars N       : Process maximum N variables (default: 100)")
        print("  --model MODEL      : Use specific model (default: gpt-4o)")
        print("  --temp TEMP        : Set temperature (default: 0.3)")
        print()
        print("Environment:")
        print("  Set OPENAI_API_KEY environment variable or create openai_api_key.txt")
        print()
        print("Examples:")
        print("  python gss_question_reviser.py gss2022_filtered.json gss2022_revised.json")
        print("  python gss_question_reviser.py input.json output.json --start-from age --max-vars 50")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    # Parse options
    start_from = None
    max_variables = 1000  # Changed default to 100
    model_id = "gpt-4o"
    temperature = 0.3
    
    i = 3
    while i < len(sys.argv):
        if sys.argv[i] == "--start-from" and i + 1 < len(sys.argv):
            start_from = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == "--max-vars" and i + 1 < len(sys.argv):
            max_variables = int(sys.argv[i + 1])
            i += 2
        elif sys.argv[i] == "--model" and i + 1 < len(sys.argv):
            model_id = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == "--temp" and i + 1 < len(sys.argv):
            temperature = float(sys.argv[i + 1])
            i += 2
        else:
            print(f"Unknown option: {sys.argv[i]}")
            sys.exit(1)
    
    # Validate input file
    if not os.path.exists(input_file):
        print(f"❌ Input file not found: {input_file}")
        sys.exit(1)
    
    # Load API key
    api_key = load_api_key()
    if not api_key:
        print("❌ OpenAI API key not found!")
        print("Set OPENAI_API_KEY environment variable or create openai_api_key.txt file")
        sys.exit(1)
    
    print("✓ API key loaded")
    
    # Create reviser and run
    try:
        reviser = GSSQuestionReviser(
            api_key=api_key,
            model_id=model_id,
            temperature=temperature
        )
        
        stats = reviser.revise_all_questions(
            input_file, 
            output_file,
            start_from=start_from,
            max_variables=max_variables
        )
        
        # Print final summary
        print(f"\n🎉 REVISION COMPLETE!")
        print(f"Total processed: {stats['total_processed']}")
        print(f"Skipped (already processed): {stats['skipped']}")
        print(f"Accepted: {stats['accepted']}")
        print(f"Rejected: {stats['rejected']}")
        print(f"Errors: {stats['errors']}")
        
        if stats['total_processed'] > 0:
            print(f"Acceptance rate: {stats['accepted']/stats['total_processed']*100:.1f}%")
        
        print(f"\nOutput saved to: {output_file}")
        print(f"Log saved to: {reviser.log_pth}")
        
    except KeyboardInterrupt:
        print("\n⚠️ Process interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error during processing: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()