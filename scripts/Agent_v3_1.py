##############################################################################################################
# Maieutic Prompting Agentic Workflow
#
# Predict the conditional probability distribution P(B | A = a) by considering different scenarios in parallel:
# P(B | A = a) = Sum_x( P(X = x | A = a) * P(B | X = x, A = a))
# X space is large when we do not use tree structures so we need to sample from it.
# However, instead of sampling from P(X | A = a) directly, we use a proposal distribution:
# Q(X | A = a) = Sum_b( P(B = b | A = a) * P(X | B = b, A = a))
# 
# X can be not that probable with A = a? 
#
# Example of X: The respondent has been friends with their most recent sexual partner for several years. They met at a mutual friend's party and bonded over shared interests in art and music. Despite identifying as gay, lesbian, or homosexual, the respondent had a one-time intimate encounter with their heterosexual friend after a long night of conversation and connection. Both parties are comfortable with their respective sexual orientations and have chosen to remain close friends after the experience.
#
# Not enough detail? We want the X to contain details like "who is male and who is female", or "who was convinced by the other side"
#
# Intuitively, for each potential target value B = b we flesh out a narrative situation X = x, evaluate B through
# these and correct based on importance weights
# w(x) = P(X = x | A = a) / Sum_b( P(B = b | A = a) * P(X = x | B = b, A = a))
# and we get this by evaluating the reciprocal:
# 1/w(x) = Sum_b( P(B = b | A = a) * P(X = x | B = b, A = a)) / P(X = x | A = a)
# by asking "Given A = a, how much does B = b makes X = x more likely?" for each b.
# 
# P(B = b | A = a) can be a deterministic sequence? Check importance sampling
# Get P(X = x | B = b, A = a) from API
#
# p(A,B) != p(B,A) for LLMs indicating p inconsistencies
# It's true that p(A=a) = \sum_b p(A=a,B=b) where the summand is the probability of the string ab and the left-hand-side is the probability of the prefix a, but \sum_b p(B=b, A=a) might get a very different estimate of p(A=a). 
#
# We can give an importance weight to x = 0?
#
# Recursively?  If we are uncertain about P(B | X = x, A = a), we can recurse and evaluate P(B | X = x, Y = y, A = a). Need a confidence estimator, which we may have to learn. Maybe assume that more thinking (recursion) always gives better results if it changes them at all, so we need to figure out by how much they change it and if it's worthwhile.
#
# We can get the w(x) from a small model
#
# 1. In the deterministic case, Sum_b( P(B = b | A = a) * P(X = x | B = b, A = a)) is known because it is just P(X = x | B = b, A = a) for a particular b that generated x, so its just the proposal distribution. We can call a small model for the denominator, which is equivalent to using the small model for P(X = x | A = a) in the orignal P(B | A = a) = Sum_x( P(X = x | A = a) * P(B | X = x, A = a)). 
#
# 2. In the non-deterministic case, we can approximate the ratio either by asking the big model verbally about the ratio (the current script, note that what we really want is a string ratio, but we ask about a "real-world" ratio, so this might be problematic) or by querying a smaller model for both the numerator and the denominator of the ratio.
#
# 3. If we propose x (get P(X = x | B = b, A = a)) from the small model as well, then the final approximation in 2 is exact. And we can compare that to the verbal approach / approximation in 2 to evaluate it.
#
# For paper: try the smaller model first, then an extension (needs approximation due to API constriants) to the bigger model to see if that improves. 
#
# Prompting: what could you learn that would make B = b more likely, (but not defintely?) ? (when generating X) Things that might cause b or might be a result of b, or are correlated for some other reasons like a common cause.
#
# 
#
#
#
# Finally we have P(B | A = a) = Sum_x( w(x) * P(B | X = x, A = a)) / Sum_x( w(x))
# When # of x is finite, this is biased
# Make additional queries, unless we change the prompting to give more information
##############################################################################################################

import json
import numpy as np
from typing import List, Dict, Tuple, Any, Optional
import sys
import re
from datetime import datetime
import math
from scipy.stats import entropy


class MaieuticPrompting:
    def __init__(self, config_path: str, log_path: str, kl_path: str, model_id: str, client):
        """Initialize the maieutic prompting system."""
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        self.log_pth = log_path
        self.kl_pth = kl_path
        self.model_ID = model_id
        self.client = client
        self.temperature = self.config['model_settings']['evaluation_temperature']
        self.scenario_temp = self.config['model_settings']['scenario_generation_temperature']
        self.total_scenarios = self.config['model_settings']['total_scenarios_per_input']
        self.LLM_budget_left = float('inf')  # Initialize with no limit
        
        # Initialize log file with header
        with open(self.log_pth, "w") as log_file:
            log_file.write(f"=== Maieutic Prompting Log - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===\n\n")
        
        # Initialize KL divergence output file
        with open(self.kl_pth, "w") as kl_file:
            pass  # Create empty file, no headers or explanations
        
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
    
    def question_answer_to_fact(self, question: str, answer: str) -> str:
        """Convert a question-answer pair to a factual statement."""
        response = self.single_LLM_call(
            self.config['prompts']['question_answer_to_fact'],
            self.client,
            "question_answer_to_fact",
            {
                "QUESTION": question,
                "ANSWER": answer
            }
        )
        
        return response.strip()
    
    def parse_probability_list(self, response: str, domain_size: int) -> List[float]:
        """Extract probability list from LLM response for non-binary estimation."""
        try:
            # Clean the response - remove any extra whitespace and normalize
            cleaned_response = response.strip()
            
            # Try to find numbers separated by commas, semicolons, or spaces
            # Look for patterns like "0.5, 0.3, 0.2" or "0.5; 0.3; 0.2"
            numbers = re.findall(r'0?\.\d+|1\.0+|0+\.0+|1|0', cleaned_response)
            
            if len(numbers) == domain_size:
                probabilities = [float(num) for num in numbers]
                
                # Normalize to sum to 1
                total = sum(probabilities)
                if total > 0:
                    probabilities = [p / total for p in probabilities]
                    return probabilities
            
            # If we couldn't parse the exact number, try to extract any valid probabilities
            elif len(numbers) > 0:
                # Take the first domain_size numbers
                probabilities = [float(numbers[i]) if i < len(numbers) else 0.0 for i in range(domain_size)]
                total = sum(probabilities)
                if total > 0:
                    probabilities = [p / total for p in probabilities]
                    return probabilities
                    
        except Exception as e:
            with open(self.log_pth, "a") as log_file:
                log_file.write(f"[WARN] Error parsing probability list: {e}\n")
        
        with open(self.log_pth, "a") as log_file:
            log_file.write(f"[WARN] Could not parse probability list from: {response}\n")
        
        # Uniform fallback
        uniform_prob = 1.0 / domain_size
        return [uniform_prob] * domain_size
    
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
    
    def should_allocate_to_option(self, b_distribution: List[float], current_allocation: List[int]) -> int:
        """Decide which option the next scenario should be allocated to."""
        best_option = 0
        best_distance = float('inf')
        
        for i in range(len(b_distribution)):
            # Try adding to option i
            alloc_test = current_allocation.copy()
            alloc_test[i] += 1
            distance = self.get_allocation_distance(b_distribution, alloc_test)
            
            if distance < best_distance:
                best_distance = distance
                best_option = i
        
        return best_option
    
    def compute_kl_divergence(self, predicted: List[float], ground_truth: List[float]) -> float:
        """Compute KL divergence KL(ground_truth || predicted) using scipy.stats.entropy."""
        # Validate inputs
        if len(predicted) != len(ground_truth):
            raise ValueError(f"Predicted length ({len(predicted)}) != ground truth length ({len(ground_truth)})")
        
        if len(ground_truth) < 2:
            raise ValueError(f"Ground truth must have at least 2 values, got {len(ground_truth)}")
            
        if abs(sum(ground_truth) - 1.0) > 1e-6:
            raise ValueError(f"Ground truth must sum to 1, got {sum(ground_truth)}")
        
        # Use scipy.stats.entropy: entropy(pk, qk) computes KL(pk || qk)
        return entropy(ground_truth, predicted)
    
    def b_value_to_fact(self, b_question: str, b_value: str) -> str:
        """Convert a B question-value pair to a factual statement."""
        response = self.single_LLM_call(
            self.config['prompts']['question_answer_to_fact'],
            self.client,
            "b_value_to_fact",
            {
                "QUESTION": b_question,
                "ANSWER": b_value
            }
        )
        
        return response.strip()
    
    def concatenate_facts(self, facts: List[str]) -> str:
        """Concatenate facts with proper grammar: comma separation with 'and that' before the last fact."""
        if len(facts) == 0:
            return ""
        elif len(facts) == 1:
            return facts[0]
        elif len(facts) == 2:
            return f"{facts[0]}, and that {facts[1]}"
        else:
            # More than 2 facts: "fact1, fact2, ..., and that lastfact"
            all_but_last = ", ".join(facts[:-1])
            return f"{all_but_last}, and that {facts[-1]}"
    
    def format_b_question_with_domain(self, b_question: str, b_domain: List[str]) -> str:
        """Format B question with domain in the required format."""
        domain_str = "; ".join(b_domain)
        return f"{b_question} [{domain_str}]"
    
    def estimate_b_directly(self, a_facts: List[str], b_question: str, b_domain: List[str]) -> List[float]:
        """Get direct estimate of P(B=each_option|A=a) for all options in domain."""
        self.temperature = self.config['model_settings']['evaluation_temperature']
        
        # Concatenate facts properly
        a_fact_string = self.concatenate_facts(a_facts)
        
        # Format B question with domain
        b_formatted = self.format_b_question_with_domain(b_question, b_domain)
        
        response = self.single_LLM_call(
            self.config['prompts']['initial_b_estimation_nonbinary'],
            self.client,
            "initial_b_estimation_nonbinary",
            {
                "A": a_fact_string,
                "B": b_formatted
            }
        )
        
        probabilities = self.parse_probability_list(response, len(b_domain))
        
        return probabilities
    
    def generate_scenario(self, a_facts: List[str], b_fact: str) -> str:
        """Generate a scenario X=x given A=a and B=b (as facts)."""
        # Use higher temperature for scenario generation
        self.temperature = self.config['model_settings']['scenario_generation_temperature']
        
        # Concatenate facts properly
        a_fact_string = self.concatenate_facts(a_facts)
        
        response = self.single_LLM_call(
            self.config['prompts']['scenario_generation'],
            self.client,
            "scenario_generation",
            {
                "A": a_fact_string,
                "B": b_fact
            }
        )
        
        # Reset temperature
        self.temperature = self.config['model_settings']['evaluation_temperature']
        
        return response.strip()
    
    def compute_importance_weight(self, a_facts: List[str], x_scenario: str, b_facts: List[str], 
                                  b_probs: List[float]) -> float:
        """Compute importance weight w(x) for a scenario using Version 2 approach."""
        # Concatenate facts properly
        a_fact_string = self.concatenate_facts(a_facts)
        
        # Compute 1/w(x) = sum_b P(B=b|A=a) * P(X=x|A=a,B=b)/P(X=x|A=a)
        reciprocal_weight = 0.0
        
        for i, (b_fact, p_b) in enumerate(zip(b_facts, b_probs)):
            # Get likelihood ratio: P(X=x|A=a,B=b) / P(X=x|A=a)
            # Ask: "If you knew B=b_fact, how many times more likely would it make X=x?"
            ratio_response = self.single_LLM_call(
                self.config['prompts']['likelihood_ratio'],
                self.client,
                "likelihood_ratio",
                {
                    "A": a_fact_string,
                    "X": x_scenario,
                    "B": b_fact
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
    
    def estimate_b_given_scenario(self, a_facts: List[str], x_scenario: str, b_question: str, 
                                  b_domain: List[str]) -> List[float]:
        """Estimate P(B=each_option|X=x,A=a) for all options."""
        # Concatenate facts properly
        a_fact_string = self.concatenate_facts(a_facts)
        
        # Format B question with domain
        b_formatted = self.format_b_question_with_domain(b_question, b_domain)
        
        response = self.single_LLM_call(
            self.config['prompts']['final_b_estimation_given_scenario_nonbinary'],
            self.client,
            "final_b_estimation_given_scenario_nonbinary",
            {
                "A": a_fact_string,
                "X": x_scenario,
                "B": b_formatted
            }
        )
        
        probabilities = self.parse_probability_list(response, len(b_domain))
        
        return probabilities
    
    def process_line(self, a_input: str, b_question: str, b_domain: List[str], ground_truth: Optional[List[float]] = None) -> Dict[str, float]:
        """Process a single input line using maieutic prompting."""
        self.b_domain = b_domain  # Store for use in parse_probability fallback
        
        # Parse A input - could be single fact or list of facts
        if a_input.startswith('[') and a_input.endswith(']'):
            # Multiple A conditions - parse as list
            a_input_clean = a_input[1:-1]  # Remove brackets
            # Split by comma but respect quotes
            a_parts = []
            current_part = ""
            in_quotes = False
            for char in a_input_clean:
                if char == '"' and not in_quotes:
                    in_quotes = True
                elif char == '"' and in_quotes:
                    in_quotes = False
                elif char == ',' and not in_quotes:
                    a_parts.append(current_part.strip())
                    current_part = ""
                    continue
                current_part += char
            if current_part.strip():
                a_parts.append(current_part.strip())
            
            # Remove quotes from each part
            a_parts = [part.strip('"').strip() for part in a_parts]
            
            # Convert each question-answer pair to fact
            a_facts = []
            for part in a_parts:
                # Split into question and answer
                question_end = part.rfind('?')
                if question_end != -1:
                    question = part[:question_end + 1].strip()
                    answer = part[question_end + 1:].strip()
                    fact = self.question_answer_to_fact(question, answer)
                    a_facts.append(fact)
                else:
                    # Fallback: treat as single fact
                    a_facts.append(part)
        else:
            # Single A condition
            question_end = a_input.rfind('?')
            if question_end != -1:
                a_question = a_input[:question_end + 1].strip()
                a_answer = a_input[question_end + 1:].strip()
                a_fact = self.question_answer_to_fact(a_question, a_answer)
                a_facts = [a_fact]
            else:
                # Fallback: treat as single fact
                a_facts = [a_input]
        
        with open(self.log_pth, "a") as log_file:
            log_file.write(f"\n{'='*100}\n")
            log_file.write(f"PROCESSING NEW INPUT\n")
            log_file.write(f"A Input: {a_input}\n")
            log_file.write(f"A Facts: {a_facts}\n")
            log_file.write(f"A Facts Concatenated: {self.concatenate_facts(a_facts)}\n")
            log_file.write(f"B Question: {b_question}\n")
            log_file.write(f"B Domain: {b_domain}\n")
            log_file.write(f"{'='*100}\n\n")
        
        # Convert B domain values to fact statements
        b_facts = []
        for b_value in b_domain:
            b_fact = self.b_value_to_fact(b_question, b_value)
            b_facts.append(b_fact)
        
        with open(self.log_pth, "a") as log_file:
            log_file.write(f"\n[B VALUE TO FACT CONVERSION]\n")
            for i, (b_value, b_fact) in enumerate(zip(b_domain, b_facts)):
                log_file.write(f"  B={b_value} -> {b_fact}\n")
            log_file.write("\n")
        
        # Step 1: Get initial B distribution estimate
        b_distribution = self.estimate_b_directly(a_facts, b_question, b_domain)
        
        # Track KL divergences
        kl_values = []
        
        # Compute initial KL divergence if ground truth is provided
        if ground_truth is not None:
            try:
                initial_kl = self.compute_kl_divergence(b_distribution, ground_truth)
                kl_values.append(initial_kl)
                
                # Write initial KL to file immediately
                with open(self.kl_pth, "a") as kl_file:
                    kl_file.write(f"{initial_kl:.6f}")
                
                with open(self.log_pth, "a") as log_file:
                    log_file.write(f"\n[INITIAL KL DIVERGENCE]: {initial_kl:.6f}\n")
            except ValueError as e:
                with open(self.log_pth, "a") as log_file:
                    log_file.write(f"[ERROR] Could not compute initial KL: {e}\n")
                return dict(zip(b_domain, b_distribution))
        
        with open(self.log_pth, "a") as log_file:
            log_file.write(f"\n[INITIAL ESTIMATE]\n")
            for i, (option, prob) in enumerate(zip(b_domain, b_distribution)):
                log_file.write(f"  P(B={option}|A) = {prob:.4f}")
                if ground_truth is not None:
                    log_file.write(f" (true: {ground_truth[i]:.4f})")
                log_file.write("\n")
            log_file.write("\n")
        
        # Step 2: Generate scenarios and compute weighted estimates
        scenarios_data = []
        current_allocation = [0] * len(b_domain)
        running_weighted_sums = [0.0] * len(b_domain)
        running_weight_sum = 0.0
        
        for i in range(self.total_scenarios):
            # Decide which B option to use for this scenario
            option_index = self.should_allocate_to_option(b_distribution, current_allocation)
            b_value = b_domain[option_index]
            b_fact = b_facts[option_index]
            current_allocation[option_index] += 1
            
            with open(self.log_pth, "a") as log_file:
                log_file.write(f"\n{'*'*60}\n")
                log_file.write(f"SCENARIO {i+1}/{self.total_scenarios}\n")
                log_file.write(f"Generating with B = {b_value}\n")
                log_file.write(f"B Fact: {b_fact}\n")
                log_file.write(f"Current allocation: {dict(zip(b_domain, current_allocation))}\n")
                log_file.write(f"{'*'*60}\n\n")
            
            # Generate scenario
            scenario = self.generate_scenario(a_facts, b_fact)
            
            with open(self.log_pth, "a") as log_file:
                log_file.write(f"\n[GENERATED SCENARIO]:\n{scenario}\n\n")
            
            # Compute importance weight
            weight = self.compute_importance_weight(a_facts, scenario, b_facts, b_distribution)
            
            # Estimate P(B=each_option|X=x,A=a)
            p_b_given_scenario = self.estimate_b_given_scenario(a_facts, scenario, b_question, b_domain)
            
            # Update running estimates
            for j in range(len(b_domain)):
                running_weighted_sums[j] += weight * p_b_given_scenario[j]
            running_weight_sum += weight
            
            current_estimates = [s / running_weight_sum if running_weight_sum > 0 else b_distribution[j] 
                               for j, s in enumerate(running_weighted_sums)]
            
            # Compute KL divergence after this scenario if ground truth is provided
            if ground_truth is not None:
                try:
                    scenario_kl = self.compute_kl_divergence(current_estimates, ground_truth)
                    kl_values.append(scenario_kl)
                    
                    # Write KL value to file immediately
                    with open(self.kl_pth, "a") as kl_file:
                        kl_file.write(f", {scenario_kl:.6f}")
                        
                except ValueError as e:
                    with open(self.log_pth, "a") as log_file:
                        log_file.write(f"[ERROR] Could not compute KL after scenario {i+1}: {e}\n")
            
            scenarios_data.append({
                "scenario": scenario,
                "b_value_used": b_value,
                "weight": weight,
                "p_b_given_scenario": p_b_given_scenario
            })
            
            with open(self.log_pth, "a") as log_file:
                log_file.write(f"\n[SCENARIO {i+1} RESULTS]\n")
                log_file.write(f"  Importance weight w(x) = {weight:.4f}\n")
                log_file.write(f"  P(B|scenario):\n")
                for k, (option, prob) in enumerate(zip(b_domain, p_b_given_scenario)):
                    log_file.write(f"    P(B={option}|scenario) = {prob:.4f}\n")
                log_file.write(f"  Current cumulative estimates:\n")
                for k, (option, est) in enumerate(zip(b_domain, current_estimates)):
                    log_file.write(f"    P(B={option}|A) = {est:.4f} (initial: {b_distribution[k]:.4f})")
                    if ground_truth is not None:
                        log_file.write(f" (true: {ground_truth[k]:.4f})")
                    log_file.write("\n")
                if ground_truth is not None:
                    log_file.write(f"  KL divergence after scenario {i+1}: {kl_values[-1]:.6f}\n")
        
        # Step 3: Compute final weighted estimate
        total_weight = sum(s["weight"] for s in scenarios_data)
        if total_weight == 0:
            with open(self.log_pth, "a") as log_file:
                log_file.write(f"\n[WARN] Total weight is 0, using initial estimate\n")
            return dict(zip(b_domain, b_distribution))
        
        weighted_probs = []
        for j in range(len(b_domain)):
            weighted_prob = sum(s["weight"] * s["p_b_given_scenario"][j] for s in scenarios_data) / total_weight
            weighted_probs.append(weighted_prob)
        
        with open(self.log_pth, "a") as log_file:
            log_file.write(f"\n{'='*60}\n")
            log_file.write(f"FINAL RESULTS\n")
            log_file.write(f"{'='*60}\n")
            log_file.write(f"Initial estimates vs Final estimates:\n")
            for i, (option, init_prob, final_prob) in enumerate(zip(b_domain, b_distribution, weighted_probs)):
                diff = final_prob - init_prob
                log_file.write(f"  P(B={option}|A): {init_prob:.4f} -> {final_prob:.4f} ({diff:+.4f})")
                if ground_truth is not None:
                    log_file.write(f" (true: {ground_truth[i]:.4f})")
                log_file.write("\n")
            log_file.write(f"Final scenario allocation: {dict(zip(b_domain, current_allocation))}\n")
            if ground_truth is not None:
                final_kl = self.compute_kl_divergence(weighted_probs, ground_truth)
                log_file.write(f"Final KL divergence: {final_kl:.6f}\n")
                log_file.write(f"KL improvement: {kl_values[0] - final_kl:.6f}\n")
            log_file.write(f"{'='*60}\n\n")
        
        # Write KL values to KL output file
        if ground_truth is not None:
            # Add newline to complete the line for this input
            with open(self.kl_pth, "a") as kl_file:
                kl_file.write("\n")
        
        return dict(zip(b_domain, weighted_probs))


def parse_input_line(line: str) -> Tuple[str, str, str, List[str], Optional[List[float]]]:
    """Parse input line with format: {B: question [domain]; A: question answer}; [ground_truth]"""
    line = line.strip()
    
    # Find key markers
    b_start = line.find("B: ")
    a_start = line.find("A: ")
    last_bracket_start = line.rfind("; [")
    
    if b_start == -1 or a_start == -1:
        raise ValueError(f"Could not find 'B: ' or 'A: ' markers in: {line}")
    
    # Extract B section (between "B: " and "A: ")
    b_section = line[b_start + 3:a_start].strip()
    if b_section.endswith(';'):
        b_section = b_section[:-1].strip()
    
    # Split B question and domain
    bracket_start = b_section.find('[')
    bracket_end = b_section.rfind(']')
    
    if bracket_start == -1 or bracket_end == -1:
        raise ValueError(f"Could not find B domain brackets in: {b_section}")
    
    b_question = b_section[:bracket_start].strip()
    b_domain_str = b_section[bracket_start + 1:bracket_end].strip()
    b_domain = [item.strip() for item in b_domain_str.split(';')]
    
    # Extract A section and ground truth
    ground_truth = None
    if last_bracket_start != -1:
        # A section is between "A: " and the last "; ["
        a_section = line[a_start + 3:last_bracket_start].strip()
        
        # Extract ground truth
        gt_end = line.rfind(']')
        if gt_end != -1:
            gt_str = line[last_bracket_start + 3:gt_end].strip()
            try:
                gt_parts = re.split(r'[;,]', gt_str)
                ground_truth = [float(part.strip()) for part in gt_parts if part.strip()]
                
                # Validate ground truth
                if len(ground_truth) < 2:
                    raise ValueError(f"Ground truth must have at least 2 values, got {len(ground_truth)}")
                
                if len(ground_truth) != len(b_domain):
                    raise ValueError(f"Ground truth length ({len(ground_truth)}) != domain size ({len(b_domain)})")
                
                total = sum(ground_truth)
                if abs(total - 1.0) > 1e-6:
                    # Normalize if needed
                    ground_truth = [gt / total for gt in ground_truth]
                    
            except (ValueError, TypeError) as e:
                raise ValueError(f"Could not parse ground truth: {gt_str}, error: {e}")
    else:
        # No ground truth, A section goes to end of line
        closing_brace = line.rfind('}')
        if closing_brace == -1:
            raise ValueError(f"Could not find closing brace in: {line}")
        a_section = line[a_start + 3:closing_brace].strip()
    
    # Parse A question and answer
    question_end = a_section.rfind('?')
    if question_end != -1:
        a_question = a_section[:question_end + 1].strip()
        a_answer = a_section[question_end + 1:].strip()
    else:
        # Fallback: assume last word/phrase is the answer
        parts = a_section.rsplit(' ', 1)
        if len(parts) == 2:
            a_question, a_answer = parts
        else:
            raise ValueError(f"Could not parse A question and answer from: {a_section}")
    
    return a_question, a_answer, b_question, b_domain, ground_truth


def main():
    """Main function to run maieutic prompting on input file."""
    # Define all file paths here
    config_path = "/Users/ruoxiliu/Desktop/Argo_research/reasoning/sense/backtracking/experiment_maieutic_nonbinary/maieutic_config.json"
    input_path = "/Users/ruoxiliu/Desktop/Argo_research/reasoning/sense/backtracking/experiment_maieutic_nonbinary/input_1.txt"
    output_path = "/Users/ruoxiliu/Desktop/Argo_research/reasoning/sense/backtracking/experiment_maieutic_nonbinary/output_1.txt"
    log_path = "/Users/ruoxiliu/Desktop/Argo_research/reasoning/sense/backtracking/experiment_maieutic_nonbinary/log_1.txt"
    kl_path = "/Users/ruoxiliu/Desktop/Argo_research/reasoning/sense/backtracking/experiment_maieutic_nonbinary/kl_output_1.txt"
    
    # Initialize OpenAI client with API key
    from openai import OpenAI
    
    # IMPORTANT: Replace with your actual OpenAI API key
    OPENAI_API_KEY = 'YOUR_API_KEY_HERE'
    
    client = OpenAI(api_key=OPENAI_API_KEY)
    
    # Initialize maieutic system
    maieutic = MaieuticPrompting(
        config_path=config_path,
        log_path=log_path,
        kl_path=kl_path,
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
            # Parse input line (now includes optional ground truth)
            a_question, a_answer, b_question, b_domain, ground_truth = parse_input_line(line)
            
            # Reconstruct a_input from parsed components
            a_input = f"{a_question} {a_answer}"
            
            # Process this input
            distribution = maieutic.process_line(a_input, b_question, b_domain, ground_truth)
            results.append(distribution)
            
        except ValueError as e:
            print(f"Error parsing line {i+1}: {e}")
            with open(log_path, "a") as log_file:
                log_file.write(f"[ERROR] Line {i+1}: {e}\n")
            # Default uniform fallback
            uniform_prob = 1.0 / len(b_domain) if 'b_domain' in locals() else 0.5
            results.append({f"option_{j}": uniform_prob for j in range(len(b_domain) if 'b_domain' in locals() else 2)})
    
    # Write output file
    with open(output_path, 'w') as f:
        for i, dist in enumerate(results):
            f.write(f"Line {i+1}: ")
            prob_strs = [f"P({option})={prob:.3f}" for option, prob in dist.items()]
            f.write(", ".join(prob_strs) + "\n")
    
    print(f"Processing complete. Results written to {output_path}")
    print(f"Detailed log available at {log_path}")
    print(f"KL divergence tracking available at {kl_path}")


if __name__ == "__main__":
    main()