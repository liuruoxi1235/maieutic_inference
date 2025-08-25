##############################################################################################################
# Maieutic Prompting
#
# Predict the conditional probability distribution P(B | A = a) by considering different scenarios in parallel:
# P(B | A = a) = Sum_x( P(X = x | A = a) * P(B | X = x, A = a))
# X space is large when we do not use tree structures so we need to sample from it.
# However, instead of sampling from P(X | A = a) directly, we use a proposal distribution:
# Q(X | A = a) = Sum_b( P(B = b | A = a) * P(X | B = b, A = a))
# 
# Intuitively, for each potential target value B = b we flush out a narrative situation X = x, evaluate B through
# these and correct based on importance weights
# w(x) = P(X = x | A = a) / Sum_b( P(B = b | A = a) * P(X = x | B = b, A = a))
# and we get this by evaluating the reciprocal:
# 1/w(x) = Sum_b( P(B = b | A = a) * P(X = x | B = b, A = a)) / P(X = x | A = a)
# by asking "Given A = a, how much does B = b makes X = x more likely?" for each b.
# 
# Finally we have P(B | A = a) = Sum_x( w(x) * P(B | X = x, A = a)) / Sum_x( w(x))
##############################################################################################################



import json
import numpy as np
from typing import List, Dict, Tuple, Any
import sys
import re
from datetime import datetime


class MaieuticPrompting:
    def __init__(self, config_path: str, log_path: str, model_id: str, client):
        """Initialize the maieutic prompting system."""
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        self.log_pth = log_path
        self.model_ID = model_id
        self.client = client
        self.temperature = self.config['model_settings']['evaluation_temperature']
        self.scenario_temp = self.config['model_settings']['scenario_generation_temperature']
        self.total_scenarios = self.config['model_settings']['total_scenarios_per_input']
        self.LLM_budget_left = float('inf')  # Initialize with no limit
        
        # Initialize log file with header
        with open(self.log_pth, "w") as log_file:
            log_file.write(f"=== Maieutic Prompting Log - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===\n\n")
        
    def single_LLM_call(self, json_prompt, client, name, replacements={}, additional_message=None):
        """Take a json styled role-name array and return the generated content, supports placeholder replacements"""
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
            log_file.write(f"[LLM CALL - {name}]\n")
            log_file.write(f"[PROMPT]:\n")
            for msg in messages:
                log_file.write(f"  {msg['role'].upper()}: {msg['content']}\n")
            log_file.write("\n")

        # Call the model
        response = client.chat.completions.create(
            model=self.model_ID,
            messages=messages,
            temperature=self.temperature
        )
        
        response_content = response.choices[0].message.content
        
        with open(self.log_pth, "a") as log_file:
            log_file.write(f"[RESPONSE]: {response_content}\n")
            log_file.write("-" * 80 + "\n\n")
        
        self.LLM_budget_left -= 1

        # Return the actual generation
        return response_content
    
    def generate_opposite_b(self, b_event: str) -> str:
        """Generate the logical negation of B event."""
        response = self.single_LLM_call(
            self.config['prompts']['generate_opposite_of_B'],
            self.client,
            "generate_opposite_of_B",
            {
                "B": b_event
            }
        )
        
        return response.strip()
    
    def parse_probability(self, response: str) -> float:
        """Extract probability from LLM response."""
        try:
            # Try to find a float in the response
            numbers = re.findall(r'0?\.\d+|1\.0|0|1', response.strip())
            if numbers:
                return float(numbers[0])
        except:
            pass
        
        with open(self.log_pth, "a") as log_file:
            log_file.write(f"[WARN] Could not parse probability from: {response}\n")
        return 0.5  # Default fallback
    
    def parse_ratio(self, response: str) -> float:
        """Extract ratio from LLM response."""
        try:
            # Try to find a number in the response
            numbers = re.findall(r'\d+\.?\d*', response.strip())
            if numbers:
                return float(numbers[0])
        except:
            pass
        
        with open(self.log_pth, "a") as log_file:
            log_file.write(f"[WARN] Could not parse ratio from: {response}\n")
        return 1.0  # Default fallback
    
    def get_allocation_distance(self, distribution: List[float], allocation: List[int]) -> float:
        """Calculate distance between distribution and normalized allocation."""
        total = sum(allocation)
        if total == 0:
            return float('inf')
        normalized_allocation = [a / total for a in allocation]
        return sum((d - a) ** 2 for d, a in zip(distribution, normalized_allocation))
    
    def should_allocate_to_yes(self, b_distribution: List[float], current_allocation: List[int]) -> bool:
        """Decide whether next scenario should be allocated to B=Yes or B=No."""
        # Try adding to Yes (index 0)
        alloc_yes = current_allocation.copy()
        alloc_yes[0] += 1
        dist_yes = self.get_allocation_distance(b_distribution, alloc_yes)
        
        # Try adding to No (index 1)
        alloc_no = current_allocation.copy()
        alloc_no[1] += 1
        dist_no = self.get_allocation_distance(b_distribution, alloc_no)
        
        return dist_yes <= dist_no
    
    def estimate_b_directly(self, a_scenario: str, b_event: str) -> float:
        """Get direct estimate of P(B=Yes|A=a)."""
        self.temperature = self.config['model_settings']['evaluation_temperature']
        
        response = self.single_LLM_call(
            self.config['prompts']['initial_b_estimation'],
            self.client,
            "initial_b_estimation",
            {
                "A": a_scenario,
                "B": b_event
            }
        )
        
        return self.parse_probability(response)
    
    def generate_scenario(self, a_scenario: str, b_event: str, b_event_negated: str, b_value: str) -> str:
        """Generate a scenario X=x given A=a and B=b."""
        # Use higher temperature for scenario generation
        self.temperature = self.config['model_settings']['scenario_generation_temperature']
        
        # Use the appropriate B based on the value
        b_to_use = b_event if b_value == "Yes" else b_event_negated
        
        response = self.single_LLM_call(
            self.config['prompts']['scenario_generation'],
            self.client,
            "scenario_generation",
            {
                "A": a_scenario,
                "B": b_to_use
            }
        )
        
        # Reset temperature
        self.temperature = self.config['model_settings']['evaluation_temperature']
        
        return response.strip()
    
    def compute_importance_weight(self, a_scenario: str, x_scenario: str, b_event: str, 
                                  b_probs: Dict[str, float]) -> float:
        """Compute importance weight w(x) for a scenario."""
        # First get P(X=x|A=a) - probability of scenario given just context
        p_x_given_a = self.parse_probability(
            self.single_LLM_call(
                self.config['prompts']['scenario_probability_given_context'],
                self.client,
                "scenario_probability",
                {
                    "A": a_scenario,
                    "B": b_event,
                    "X": x_scenario
                }
            )
        )
        
        # Compute 1/w(x) = sum_b P(B=b|A=a) * P(X=x|A=a,B=b)/P(X=x|A=a)
        reciprocal_weight = 0.0
        
        for b_value, p_b in b_probs.items():
            # Get likelihood ratio: P(X=x|A=a,B=b) / P(X=x|A=a)
            ratio_response = self.single_LLM_call(
                self.config['prompts']['likelihood_ratio'],
                self.client,
                "likelihood_ratio",
                {
                    "A": a_scenario,
                    "X": x_scenario,
                    "B": b_event
                }
            )
            ratio = self.parse_ratio(ratio_response)
            
            reciprocal_weight += p_b * ratio
        
        # Avoid division by zero
        if reciprocal_weight == 0:
            with open(self.log_pth, "a") as log_file:
                log_file.write(f"[WARN] Reciprocal weight is 0, using default weight of 1.0\n")
            return 1.0
        
        return 1.0 / reciprocal_weight
    
    def estimate_b_given_scenario(self, a_scenario: str, x_scenario: str, b_event: str) -> float:
        """Estimate P(B=Yes|X=x,A=a)."""
        response = self.single_LLM_call(
            self.config['prompts']['final_b_estimation_given_scenario'],
            self.client,
            "final_b_estimation",
            {
                "A": a_scenario,
                "X": x_scenario,
                "B": b_event
            }
        )
        
        return self.parse_probability(response)
    
    def process_line(self, a_scenario: str, b_event: str) -> Dict[str, float]:
        """Process a single input line using maieutic prompting."""
        with open(self.log_pth, "a") as log_file:
            log_file.write(f"\n{'='*100}\n")
            log_file.write(f"PROCESSING NEW INPUT\n")
            log_file.write(f"B Variable: {b_event}\n")
            log_file.write(f"A Scenario: {a_scenario}\n")
            log_file.write(f"{'='*100}\n\n")
        
        # Generate the negation of B once at the beginning
        b_event_negated = self.generate_opposite_b(b_event)
        
        with open(self.log_pth, "a") as log_file:
            log_file.write(f"[NEGATION] B negated: {b_event_negated}\n\n")
        
        # Step 1: Get initial B distribution estimate
        p_b_yes_initial = self.estimate_b_directly(a_scenario, b_event)
        b_distribution = [p_b_yes_initial, 1 - p_b_yes_initial]
        b_probs = {"Yes": p_b_yes_initial, "No": 1 - p_b_yes_initial}
        
        with open(self.log_pth, "a") as log_file:
            log_file.write(f"\n[INITIAL ESTIMATE] P(B=Yes|A) = {p_b_yes_initial:.4f}\n\n")
        
        # Step 2: Generate scenarios and compute weighted estimates
        scenarios_data = []
        current_allocation = [0, 0]  # [Yes count, No count]
        running_weighted_sum = 0.0
        running_weight_sum = 0.0
        
        for i in range(self.total_scenarios):
            # Decide which B value to use for this scenario
            use_yes = self.should_allocate_to_yes(b_distribution, current_allocation)
            b_value = "Yes" if use_yes else "No"
            current_allocation[0 if use_yes else 1] += 1
            
            with open(self.log_pth, "a") as log_file:
                log_file.write(f"\n{'*'*60}\n")
                log_file.write(f"SCENARIO {i+1}/{self.total_scenarios}\n")
                log_file.write(f"Generating with B = {b_value}\n")
                log_file.write(f"Current allocation: Yes={current_allocation[0]}, No={current_allocation[1]}\n")
                log_file.write(f"{'*'*60}\n\n")
            
            # Generate scenario
            scenario = self.generate_scenario(a_scenario, b_event, b_event_negated, b_value)
            
            with open(self.log_pth, "a") as log_file:
                log_file.write(f"\n[GENERATED SCENARIO]:\n{scenario}\n\n")
            
            # Compute importance weight
            weight = self.compute_importance_weight(a_scenario, scenario, b_event, b_probs)
            
            # Estimate P(B=Yes|X=x,A=a)
            p_b_yes_given_scenario = self.estimate_b_given_scenario(a_scenario, scenario, b_event)
            
            # Update running estimates
            running_weighted_sum += weight * p_b_yes_given_scenario
            running_weight_sum += weight
            current_estimate = running_weighted_sum / running_weight_sum if running_weight_sum > 0 else p_b_yes_initial
            
            scenarios_data.append({
                "scenario": scenario,
                "b_value_used": b_value,
                "weight": weight,
                "p_b_yes": p_b_yes_given_scenario
            })
            
            with open(self.log_pth, "a") as log_file:
                log_file.write(f"\n[SCENARIO {i+1} RESULTS]\n")
                log_file.write(f"  Importance weight w(x) = {weight:.4f}\n")
                log_file.write(f"  P(B=Yes|scenario) = {p_b_yes_given_scenario:.4f}\n")
                log_file.write(f"  Current cumulative estimate: P(B=Yes|A) = {current_estimate:.4f}\n")
                log_file.write(f"  (Initial estimate was: {p_b_yes_initial:.4f})\n")
        
        # Step 3: Compute final weighted estimate
        total_weight = sum(s["weight"] for s in scenarios_data)
        if total_weight == 0:
            with open(self.log_pth, "a") as log_file:
                log_file.write(f"\n[WARN] Total weight is 0, using initial estimate\n")
            return {"Yes": p_b_yes_initial, "No": 1 - p_b_yes_initial}
        
        weighted_p_b_yes = sum(s["weight"] * s["p_b_yes"] for s in scenarios_data) / total_weight
        
        with open(self.log_pth, "a") as log_file:
            log_file.write(f"\n{'='*60}\n")
            log_file.write(f"FINAL RESULTS\n")
            log_file.write(f"{'='*60}\n")
            log_file.write(f"Initial estimate: P(B=Yes|A) = {p_b_yes_initial:.4f}\n")
            log_file.write(f"Final estimate:   P(B=Yes|A) = {weighted_p_b_yes:.4f}\n")
            log_file.write(f"Difference: {weighted_p_b_yes - p_b_yes_initial:+.4f}\n")
            log_file.write(f"Final scenario allocation: Yes={current_allocation[0]}, No={current_allocation[1]}\n")
            log_file.write(f"{'='*60}\n\n")
        
        return {"Yes": weighted_p_b_yes, "No": 1 - weighted_p_b_yes}


def parse_input_line(line: str) -> Tuple[str, str]:
    """Parse input line of format {B: [B variable]; A: [A scenario]}"""
    # Match the pattern {B: ...; A: ...}
    pattern = r'\{B:\s*([^;]+);\s*A:\s*([^}]+)\}'
    match = re.match(pattern, line.strip())
    
    if match:
        b_variable = match.group(1).strip()
        a_scenario = match.group(2).strip()
        return a_scenario, b_variable
    else:
        raise ValueError(f"Invalid input format: {line}")


def main():
    """Main function to run maieutic prompting on input file."""
    # Define all file paths here
    config_path = "/Users/ruoxiliu/Desktop/Argo_research/reasoning/sense/backtracking/experiment_maieutic/maieutic_config.json"
    input_path = "/Users/ruoxiliu/Desktop/Argo_research/reasoning/sense/backtracking/experiment_maieutic/input.txt"
    output_path = "/Users/ruoxiliu/Desktop/Argo_research/reasoning/sense/backtracking/experiment_maieutic/output.txt"
    log_path = "/Users/ruoxiliu/Desktop/Argo_research/reasoning/sense/backtracking/experiment_maieutic/log_3.txt"
    
    # Initialize OpenAI client with API key
    from openai import OpenAI
    
    # IMPORTANT: Replace with your actual OpenAI API key
    OPENAI_API_KEY = "YOUR_OPENAI_API_KEY"
    
    client = OpenAI(api_key=OPENAI_API_KEY)
    
    # Initialize maieutic system
    maieutic = MaieuticPrompting(
        config_path=config_path,
        log_path=log_path,
        model_id="gpt-4o-2024-08-06",
        client=client
    )
    
    # Process input file
    with open(input_path, 'r') as f:
        lines = f.readlines()
    
    results = []
    
    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
        
        try:
            # Parse input line
            a_scenario, b_event = parse_input_line(line)
            
            # Process this input
            distribution = maieutic.process_line(a_scenario, b_event)
            results.append(distribution)
            
        except ValueError as e:
            print(f"Error parsing line {i+1}: {e}")
            with open(log_path, "a") as log_file:
                log_file.write(f"[ERROR] Line {i+1}: {e}\n")
            results.append({"Yes": 0.5, "No": 0.5})  # Default fallback
    
    # Write output file
    with open(output_path, 'w') as f:
        for i, dist in enumerate(results):
            f.write(f"Line {i+1}: P(Yes)={dist['Yes']:.3f}, P(No)={dist['No']:.3f}\n")
    
    print(f"Processing complete. Results written to {output_path}")
    print(f"Detailed log available at {log_path}")


if __name__ == "__main__":
    main()