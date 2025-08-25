##############################################################################################################
# Maieutic Prompting with Multiple Importance Sampling (MIS)
#
# Quick start:
# 1. Define your configuration in a JSON file (see example config files), then define input lines in a single file (also see examples).
# 2. Put these two files in a single folder, then set input_folder in the beginning of main to that folder path.
# 3. Set up your OpenAI API key in the the beginning of main in script below.
#
# Estimator: Σᵢ Σⱼ P(Xᵢⱼ|A=a)P(B=b|Xᵢⱼ,A=a) / Σₖ nₖP(Xᵢⱼ|A=a,B=bₖ)
#
# Modes:
# - "verbal": Use OpenAI to get likelihood ratios and P(B|X,A)
# - "verbal_small": Use HuggingFace for scenario generation and likelihood ratios, OpenAI for P(B|X,A)
# - "string": Use huggingface models for individual probabilities, OpenAI for P(B|X,A)
# - "string_scaled": Same as string mode but with likelihood ratio scaling normalization
#
# Allocation methods:
# - "online": Adaptive allocation based on current estimates
# - "prealloc": Fixed allocation based on smoothed initial distribution
# - "prealloc_fill_first": Fill each option once, then use optimal allocation
#
##############################################################################################################

import json
import numpy as np
from typing import List, Dict, Tuple, Any, Optional
import sys
import re
from datetime import datetime
import math
from scipy.stats import entropy
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
from openai import OpenAI
from vllm import LLM, SamplingParams

class MaieuticPromptingMIS:
    def __init__(self, config_path: str, log_path: str, kl_path: str, model_id: str, client, weights_path: str = None, ess_path: str = None):
        """Initialize the maieutic prompting system with MIS."""
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
        
        # Get mode from config
        self.mode = self.config['model_settings'].get('mode', 'verbal')

        # Get allocation mode from config (default to 'online' for backward compatibility)
        self.allocation_mode = self.config['model_settings'].get('allocation', 'online')
        
        # Create output directory if it doesn't exist 
        base_dir = os.path.dirname(log_path)
        os.makedirs(base_dir, exist_ok=True)
        
        # Set output file paths
        if weights_path is None:
            self.weights_pth = os.path.join(base_dir, "weights_output.txt")
        else:
            self.weights_pth = weights_path

        if ess_path is None:
            self.ess_pth = os.path.join(base_dir, "ESS_output.txt")
        else:
            self.ess_pth = ess_path

        self.kl_unweighted_pth = os.path.join(base_dir, "kl_unweighted_output.txt")
        self.kl_flattened_pth = os.path.join(base_dir, "kl_flattened_output.txt")
        
        # Initialize small model if in string, string_scaled, or verbal_small mode
        self.small_model = None
        self.small_tokenizer = None
        if self.mode in ['string', 'string_scaled', 'verbal_small']:
            self._initialize_small_model()
        
        # Initialize log file with header
        with open(self.log_pth, "w") as log_file:
            log_file.write(f"=== Maieutic Prompting MIS Log - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===\n")
            log_file.write(f"Mode: {self.mode}\n")
            log_file.write(f"Allocation: {self.allocation_mode}\n\n")
        
        # Initialize KL divergence output file
        with open(self.kl_pth, "w") as kl_file:
            pass  # Create empty file
        
        # Initialize additional KL divergence output files
        with open(self.kl_unweighted_pth, "w") as kl_file:
            pass  # Create empty file

        with open(self.kl_flattened_pth, "w") as kl_file:
            pass  # Create empty file
        
        # Initialize weights output file
        with open(self.weights_pth, "w") as weights_file:
            pass  # Create empty file
        
        # Initialize ESS output file
        with open(self.ess_pth, "w") as ess_file:
            pass  # Create empty file
    
    def _initialize_small_model(self):
        """Initialize the vLLM model for token probability computation and text generation."""
        try:
            
            model_name = self.config['model_settings']['small_model']['model_name']
            
            with open(self.log_pth, "a") as log_file:
                log_file.write(f"[INIT] Loading vLLM model: {model_name}\n")
            
            # Initialize vLLM model
            self.vllm_model = LLM(
                model=model_name,
                max_model_len=32768,
                gpu_memory_utilization=0.95
            )
            
            # Store scenario generation temperature for vLLM use
            self.scenario_temp = self.config['model_settings']['scenario_generation_temperature']
            
            # Check if we're using non-instruction-tuned prompts
            self.use_non_instruction_prompts = self.config['model_settings']['small_model'].get('non_instruction_tuned', True)
            
            with open(self.log_pth, "a") as log_file:
                log_file.write(f"[INIT] vLLM model loaded successfully\n")
                log_file.write(f"[INIT] Using {'non-instruction-tuned' if self.use_non_instruction_prompts else 'instruction-tuned'} prompts\n")
                log_file.write(f"[INIT] Model ready for probability computation and text generation\n\n")
            
            self.small_model = self.vllm_model  # For compatibility
            self.small_tokenizer = None  # vLLM handles tokenization internally
                        
        except Exception as e:
            with open(self.log_pth, "a") as log_file:
                log_file.write(f"[ERROR] Failed to initialize vLLM model: {e}\n")
            raise RuntimeError(f"Failed to initialize vLLM model: {e}")
    
    def get_token_log_probability(self, prompt: str, completion: str) -> float:
        """
        Calculate the log probability of generating the completion given the prompt using vLLM.
        Returns the sum of log probabilities for all tokens in the completion.
        Note: Adds "Fleshed situation: " prefix to completion since it was parsed out during generation.
        """
        if self.small_model is None:
            raise RuntimeError("vLLM model not initialized")
        
        try:

            # Add the "Fleshed situation: " prefix back to the completion
            # since it was removed during parsing but is needed for accurate probability calculation
            completion_with_prefix = "Fleshed situation: " + completion
            
            # Combine prompt and completion
            full_text = prompt + completion_with_prefix
            
            # Set up sampling params to get log probabilities
            sampling_params = SamplingParams(
                temperature=0.0,  # Deterministic for probability calculation
                max_tokens=1,  # We're not generating, just scoring
                prompt_logprobs=0,  # Get log probs for all prompt tokens
                logprobs=0  # Get log probs for generated tokens
            )
            
            # Get output with log probabilities
            outputs = self.vllm_model.generate([full_text], sampling_params)
            
            if outputs and len(outputs) > 0:
                output = outputs[0]
                
                # Get prompt logprobs
                prompt_logprobs = output.prompt_logprobs
                
                if prompt_logprobs:
                    # Find where the completion starts (approximate by counting tokens)
                    # Since we don't have direct access to tokenizer, we estimate
                    # based on the ratio of prompt to full text length
                    prompt_ratio = len(prompt) / len(full_text)
                    prompt_token_count = int(len(prompt_logprobs) * prompt_ratio)
                    
                    # Sum log probabilities for completion tokens
                    total_log_prob = 0.0
                    for i in range(prompt_token_count, len(prompt_logprobs)):
                        if prompt_logprobs[i] is not None:
                            # Get the log prob of the most likely token (which should be the actual token)
                            if isinstance(prompt_logprobs[i], dict) and len(prompt_logprobs[i]) > 0:
                                # Get the first (most likely) token's log prob
                                token_log_prob = list(prompt_logprobs[i].values())[0].logprob
                                total_log_prob += token_log_prob
                    
                    return total_log_prob
                
            # Fallback if we couldn't get logprobs
            return -10.0  # Arbitrary low value
                
        except Exception as e:
            with open(self.log_pth, "a") as log_file:
                log_file.write(f"[ERROR] Failed to compute token log probability with vLLM: {e}\n")
            raise RuntimeError(f"Failed to compute token log probability: {e}")
    
    def compute_effective_sample_size(self, normalized_weights: List[float]) -> float:
        """
        Compute the Effective Sample Size (ESS) from normalized weights.
        ESS = (Σw)² / Σ(w²) where w are the normalized weights.
        Since weights are normalized, Σw = 1, so ESS = 1 / Σ(w²).
        """
        if not normalized_weights or len(normalized_weights) == 0:
            return 0.0
        
        # Calculate sum of squared weights
        sum_squared_weights = sum(w * w for w in normalized_weights)
        
        # Avoid division by zero
        if sum_squared_weights == 0:
            return 0.0
        
        # ESS = 1 / Σ(w²) for normalized weights
        ess = 1.0 / sum_squared_weights
        
        return ess
    
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
    
    def smooth_distribution(self, distribution: List[float]) -> List[float]:
        """Smooth a probability distribution for preallocation."""
        # Simple additive smoothing with a small epsilon
        epsilon = 0.05
        smoothed = [p + epsilon for p in distribution]
        
        # Renormalize
        total = sum(smoothed)
        smoothed = [p / total for p in smoothed]
        
        return smoothed
    
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
    
    def should_allocate_to_option_fill_first(self, b_distribution: List[float], current_allocation: List[int], scenario_index: int) -> int:
        """
        Decide allocation with fill-first strategy:
        Phase 1: Give one sample to each B value in order
        Phase 2: Use normal should_allocate_to_option logic
        """
        num_options = len(b_distribution)
        
        # Phase 1: Fill each option once (round-robin)
        if scenario_index < num_options:
            return scenario_index
        
        # Phase 2: Use normal allocation strategy
        return self.should_allocate_to_option(b_distribution, current_allocation)
    
    def get_allocation_method_name(self) -> str:
        """Get human-readable name for current allocation method."""
        method_names = {
            'online': 'Online (adaptive)',
            'prealloc': 'Pre-allocation (fixed)',
            'prealloc_fill_first': 'Pre-allocation with fill-first'
        }
        return method_names.get(self.allocation_mode, self.allocation_mode)
    
    def compute_kl_divergence_variants(self, predicted: List[float], ground_truth: List[float]) -> Tuple[float, float, float]:
        """
        Compute three KL divergence variants:
        1. Normal KL: Σᵢ pᵢ * log(pᵢ / qᵢ) (standard)
        2. Unweighted KL: Σᵢ |log(pᵢ) - log(qᵢ)| (absolute difference of logs)
        3. Flattened KL: Σᵢ pᵢ_flattened * |log(pᵢ) - log(qᵢ)| (flattened weighting with absolute difference)
        
        Returns: (normal_kl, unweighted_kl, flattened_kl)
        """
        # Validate inputs
        if len(predicted) != len(ground_truth):
            raise ValueError(f"Predicted length ({len(predicted)}) != ground truth length ({len(ground_truth)})")
        
        if len(ground_truth) < 2:
            raise ValueError(f"Ground truth must have at least 2 values, got {len(ground_truth)}")
        
        # Add small epsilon to avoid log(0) issues
        epsilon = 1e-10
        
        # Smooth and renormalize both distributions
        predicted_smooth = [p + epsilon for p in predicted]
        ground_truth_smooth = [p + epsilon for p in ground_truth]
        
        predicted_sum = sum(predicted_smooth)
        ground_truth_sum = sum(ground_truth_smooth)
        
        predicted_smooth = [p / predicted_sum for p in predicted_smooth]
        ground_truth_smooth = [p / ground_truth_sum for p in ground_truth_smooth]
        
        # 1. Normal KL (current implementation)
        normal_kl = entropy(ground_truth_smooth, predicted_smooth)
        
        # 2. Unweighted KL: Σᵢ |log(pᵢ) - log(qᵢ)|
        unweighted_kl = sum(abs(math.log(ground_truth_smooth[i]) - math.log(predicted_smooth[i])) 
                           for i in range(len(ground_truth_smooth)))
        
        # 3. Flattened KL: Flatten ground truth, then use as weights
        # Create flattened version of ground truth (more uniform)
        flattening_factor = 0.3  # Adjustable parameter (0 = no flattening, 1 = uniform)
        uniform_dist = [1.0 / len(ground_truth) for _ in ground_truth]
        ground_truth_flattened = [
            (1 - flattening_factor) * ground_truth_smooth[i] + flattening_factor * uniform_dist[i]
            for i in range(len(ground_truth_smooth))
        ]
        
        # Renormalize flattened distribution
        flattened_sum = sum(ground_truth_flattened)
        ground_truth_flattened = [p / flattened_sum for p in ground_truth_flattened]
        
        # Compute flattened KL: Σᵢ pᵢ_flattened * |log(pᵢ) - log(qᵢ)|
        flattened_kl = sum(ground_truth_flattened[i] * abs(math.log(ground_truth_smooth[i]) - math.log(predicted_smooth[i]))
                          for i in range(len(ground_truth_smooth)))
        
        return normal_kl, unweighted_kl, flattened_kl
    
    def compute_kl_divergence(self, predicted: List[float], ground_truth: List[float]) -> float:
        """Compute standard KL divergence (for backward compatibility)."""
        normal_kl, _, _ = self.compute_kl_divergence_variants(predicted, ground_truth)
        return normal_kl
    
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
        # Store original temperature for restoration
        original_temp = self.temperature
        self.temperature = self.config['model_settings']['scenario_generation_temperature']
        
        try:
            if self.mode in ['string', 'string_scaled', 'verbal_small']:
                # Use HuggingFace model for scenario generation
                result = self.generate_scenario_with_small_model(a_facts, b_fact)
            else:  # verbal mode
                # Use OpenAI for scenario generation (existing behavior)
                result = self.single_LLM_call(
                    self.config['prompts']['scenario_generation'],
                    self.client,
                    "scenario_generation",
                    {
                        "A": self.concatenate_facts(a_facts),
                        "B": b_fact
                    }
                )
        finally:
            # Always restore original temperature
            self.temperature = original_temp
        
        return result.strip()

    def generate_scenario_with_small_model(self, a_facts: List[str], b_fact: str) -> str:
        """Generate a scenario using vLLM with retry logic to ensure proper format."""
        if self.small_model is None:
            raise RuntimeError("vLLM model not initialized")
        
        try:
            from vllm import SamplingParams
            
            # Use non-instruction-tuned prompt if configured
            if self.use_non_instruction_prompts:
                # Use the non-instruction-tuned version
                prompt_template = self.config['prompts']['scenario_generation_non_instruction_tuned'][0][0]
                prompt_text = prompt_template.replace("{B}", b_fact).replace("{A}", self.concatenate_facts(a_facts))
            else:
                # Build prompt from the instruction-tuned template
                scenario_prompt = self.config['prompts']['scenario_generation']
                prompt_text = ""
                for role, content in scenario_prompt:
                    content_filled = content.replace("{A}", self.concatenate_facts(a_facts))
                    content_filled = content_filled.replace("{B}", b_fact)
                    
                    if role == "system":
                        prompt_text += f"System: {content_filled}\n"
                    elif role == "user":
                        prompt_text += f"User: {content_filled}\nAssistant: "
                    elif role == "assistant":
                        prompt_text += f"{content_filled}\n"
            
            # Set up sampling parameters
            sampling_params = SamplingParams(
                temperature=self.scenario_temp,
                max_tokens=200, 
                top_p=0.95
            )
            
            # Expected prefixes that indicate proper format
            expected_prefixes = ["Fleshed situation: "]
            
            # Retry logic: attempt up to 10 times
            max_attempts = 10
            for attempt in range(max_attempts):
                # Generate with vLLM
                outputs = self.vllm_model.generate([prompt_text], sampling_params)
                
                if outputs and len(outputs) > 0:
                    generated_text = outputs[0].outputs[0].text.strip()
                    
                    # Parse result: only consider text up to first line break
                    first_line = generated_text.split('\n')[0].strip()
                    
                    # Check if the generated text starts with expected prefix
                    has_expected_prefix = any(first_line.startswith(prefix) for prefix in expected_prefixes)
                    
                    if has_expected_prefix:
                        # Success! Remove the prefix and return
                        parsed_result = first_line
                        
                        # Remove the "Fleshed situation:" or "Fleshed situation: " prefix
                        for prefix in expected_prefixes:
                            if parsed_result.startswith(prefix):
                                parsed_result = parsed_result[len(prefix):].strip()
                                break
                        
                        # Log successful generation
                        with open(self.log_pth, "a") as log_file:
                            log_file.write(f"[vLLM SCENARIO GENERATION - SUCCESS on attempt {attempt + 1}]\n")
                            log_file.write(f"[PROMPT]: {prompt_text}\n")
                            log_file.write(f"[RAW RESPONSE]: {generated_text}\n")
                            log_file.write(f"[FIRST LINE]: {first_line}\n")
                            log_file.write(f"[PARSED RESPONSE]: {parsed_result}\n")
                            log_file.write("-" * 80 + "\n\n")
                        
                        return parsed_result
                    else:
                        # Failed attempt - log and continue
                        with open(self.log_pth, "a") as log_file:
                            log_file.write(f"[vLLM SCENARIO GENERATION - RETRY {attempt + 1}/{max_attempts}]\n")
                            log_file.write(f"[REASON]: Generated text does not start with expected prefix\n")
                            log_file.write(f"[EXPECTED]: One of {expected_prefixes}\n")
                            log_file.write(f"[ACTUAL FIRST LINE]: {first_line}\n")
                            log_file.write(f"[RAW RESPONSE]: {generated_text}\n")
                            log_file.write("-" * 40 + "\n")
            
            # If we reach here, all attempts failed
            with open(self.log_pth, "a") as log_file:
                log_file.write(f"[vLLM SCENARIO GENERATION - FAILED AFTER {max_attempts} ATTEMPTS]\n")
                log_file.write(f"[ERROR]: Unable to generate text with proper 'Fleshed situation:' prefix\n")
                log_file.write(f"[PROMPT]: {prompt_text}\n")
                log_file.write("-" * 80 + "\n\n")
            
            # Fallback: return a generic scenario with proper format
            return "Unable to generate scenario with proper format after 10 attempts"
            
        except Exception as e:
            with open(self.log_pth, "a") as log_file:
                log_file.write(f"[ERROR] vLLM scenario generation failed: {e}\n")
            raise RuntimeError(f"vLLM scenario generation failed: {e}")
    
    def get_likelihood_ratio_verbal(self, a_facts: List[str], x_scenario: str, b_fact: str) -> float:
        """Get likelihood ratio P(X|A,B)/P(X|A) using verbal prompting with OpenAI."""
        a_fact_string = self.concatenate_facts(a_facts)
        
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
        
        return self.parse_ratio(ratio_response)
    
    def get_likelihood_ratio_with_small_model(self, a_facts: List[str], x_scenario: str, b_fact: str) -> float:
        """Get likelihood ratio P(X|A,B)/P(X|A) using vLLM."""
        if self.small_model is None:
            raise RuntimeError("vLLM model not initialized")
        
        try:
            from vllm import SamplingParams
            import re
            
            # Use non-instruction-tuned prompt if configured
            if self.use_non_instruction_prompts:
                # Use the non-instruction-tuned version
                prompt_template = self.config['prompts']['likelihood_ratio_non_instruction_tuned'][0][0]
                prompt_text = prompt_template.replace("{A}", self.concatenate_facts(a_facts))
                prompt_text = prompt_text.replace("{X}", x_scenario)
                prompt_text = prompt_text.replace("{B}", b_fact)
            else:
                # Build prompt from the instruction-tuned template
                ratio_prompt = self.config['prompts']['likelihood_ratio']
                prompt_text = ""
                for role, content in ratio_prompt:
                    content_filled = content.replace("{A}", self.concatenate_facts(a_facts))
                    content_filled = content_filled.replace("{X}", x_scenario)
                    content_filled = content_filled.replace("{B}", b_fact)
                    
                    if role == "system":
                        prompt_text += f"System: {content_filled}\n"
                    elif role == "user":
                        prompt_text += f"User: {content_filled}\nAssistant: "
                    elif role == "assistant":
                        prompt_text += f"{content_filled}\n"
            
            # Clean up consecutive dots: replace 2 or more consecutive dots with a single dot
            prompt_text_cleaned = re.sub(r'\.{2,}', '.', prompt_text)
            
            # Log the cleaning if any changes were made
            if prompt_text != prompt_text_cleaned:
                with open(self.log_pth, "a") as log_file:
                    log_file.write(f"[PROMPT CLEANING] Removed consecutive dots from prompt\n")
                    log_file.write(f"[BEFORE]: {repr(prompt_text)}\n")
                    log_file.write(f"[AFTER]: {repr(prompt_text_cleaned)}\n")
                    log_file.write("-" * 40 + "\n")
            
            # Set up sampling parameters
            sampling_params = SamplingParams(
                temperature=self.temperature,  # Use evaluation temperature
                max_tokens=50,  # Shorter response for likelihood ratios
                top_p=0.95
            )
            
            # Generate with vLLM using the cleaned prompt
            outputs = self.vllm_model.generate([prompt_text_cleaned], sampling_params)
            
            if outputs and len(outputs) > 0:
                response = outputs[0].outputs[0].text.strip()
                
                # Parse the ratio using existing parsing logic
                ratio = self.parse_ratio(response)
                
                # Log the generation
                with open(self.log_pth, "a") as log_file:
                    log_file.write(f"[vLLM LIKELIHOOD RATIO]\n")
                    log_file.write(f"[PROMPT]: {prompt_text_cleaned}\n")
                    log_file.write(f"[RESPONSE]: {response}\n")
                    log_file.write(f"[PARSED RATIO]: {ratio}\n")
                    log_file.write("-" * 80 + "\n\n")
                
                return ratio
            
            # Fallback
            return 1.0
            
        except Exception as e:
            with open(self.log_pth, "a") as log_file:
                log_file.write(f"[ERROR] vLLM likelihood ratio failed: {e}\n")
            # Return default ratio on error
            return 1.0
    
    def get_likelihood_ratio_verbal_mode_aware(self, a_facts: List[str], x_scenario: str, b_fact: str) -> float:
        """Get likelihood ratio with mode-aware routing."""
        if self.mode == 'verbal_small':
            return self.get_likelihood_ratio_with_small_model(a_facts, x_scenario, b_fact)
        else:  # verbal mode (original behavior)
            return self.get_likelihood_ratio_verbal(a_facts, x_scenario, b_fact)
    
    def get_probabilities_string(self, a_facts: List[str], x_scenario: str, b_facts: List[str]) -> Tuple[float, List[float]]:
        """Get P(X|A) and P(X|A,B=b_k) for all k using vLLM."""
        # Get P(X|A)
        if self.use_non_instruction_prompts:
            # Use non-instruction-tuned prompt
            prompt_template = self.config['prompts']['scenario_generation_no_b_non_instruction_tuned'][0][0]
            prompt_a_only = prompt_template.replace("{A}", self.concatenate_facts(a_facts))
        else:
            # Use instruction-tuned prompt
            scenario_prompt_no_b = self.config['prompts']['scenario_generation_no_b']
            messages_a_only = []
            for role, content in scenario_prompt_no_b:
                content_filled = content.replace("{A}", self.concatenate_facts(a_facts))
                messages_a_only.append({"role": role, "content": content_filled})
            
            prompt_a_only = ""
            for msg in messages_a_only:
                if msg["role"] == "system":
                    prompt_a_only += f"System: {msg['content']}\n"
                elif msg["role"] == "assistant":
                    prompt_a_only += f"{msg['content']}\n"
        
        log_prob_x_given_a = self.get_token_log_probability(prompt_a_only, x_scenario)
        prob_x_given_a = math.exp(log_prob_x_given_a)
        
        # Get P(X|A,B=b_k) for each k
        prob_x_given_a_and_bk = []
        
        for k, b_fact in enumerate(b_facts):
            if self.use_non_instruction_prompts:
                # Use non-instruction-tuned prompt
                prompt_template = self.config['prompts']['scenario_generation_non_instruction_tuned'][0][0]
                prompt_a_and_b = prompt_template.replace("{A}", self.concatenate_facts(a_facts))
                prompt_a_and_b = prompt_a_and_b.replace("{B}", b_fact)
            else:
                # Use instruction-tuned prompt
                scenario_prompt_with_b = self.config['prompts']['scenario_generation']
                messages_a_and_b = []
                for role, content in scenario_prompt_with_b:
                    content_filled = content.replace("{A}", self.concatenate_facts(a_facts))
                    content_filled = content_filled.replace("{B}", b_fact)
                    messages_a_and_b.append({"role": role, "content": content_filled})
                
                prompt_a_and_b = ""
                for msg in messages_a_and_b:
                    if msg["role"] == "system":
                        prompt_a_and_b += f"System: {msg['content']}\n"
                    elif msg["role"] == "user":
                        prompt_a_and_b += f"User: {msg['content']}\nAssistant: "
                    elif msg["role"] == "assistant":
                        prompt_a_and_b += f"{msg['content']}\n"
            
            log_prob = self.get_token_log_probability(prompt_a_and_b, x_scenario)
            prob_x_given_a_and_bk.append(math.exp(log_prob))
        
        return prob_x_given_a, prob_x_given_a_and_bk
    
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
        """Process a single input line using maieutic prompting with MIS."""
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
            log_file.write(f"Mode: {self.mode}\n")
            log_file.write(f"Allocation: {self.allocation_mode}\n")
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
        
        # Track all KL divergence variants
        kl_values = []
        kl_unweighted_values = []
        kl_flattened_values = []
        
        # Track unnormalized total weights
        weight_values = []
        
        # Track ESS values
        ess_values = []
        
        # Track which B values have been sampled
        b_values_sampled = set()
        s_full_index = None  # Index of the scenario after which all B values have been sampled
        
        # Compute initial KL divergences if ground truth is provided
        if ground_truth is not None:
            try:
                initial_kl, initial_kl_unweighted, initial_kl_flattened = self.compute_kl_divergence_variants(b_distribution, ground_truth)
                kl_values.append(initial_kl)
                kl_unweighted_values.append(initial_kl_unweighted)
                kl_flattened_values.append(initial_kl_flattened)
                
                # Write initial KL values to files immediately
                with open(self.kl_pth, "a") as kl_file:
                    kl_file.write(f"{initial_kl:.6f}")
                
                with open(self.kl_unweighted_pth, "a") as kl_file:
                    kl_file.write(f"{initial_kl_unweighted:.6f}")
                
                with open(self.kl_flattened_pth, "a") as kl_file:
                    kl_file.write(f"{initial_kl_flattened:.6f}")
                
                with open(self.log_pth, "a") as log_file:
                    log_file.write(f"\n[INITIAL KL DIVERGENCES]:\n")
                    log_file.write(f"  Normal KL: {initial_kl:.6f}\n")
                    log_file.write(f"  Unweighted KL: {initial_kl_unweighted:.6f}\n")
                    log_file.write(f"  Flattened KL: {initial_kl_flattened:.6f}\n")
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
        
        # Determine allocation vector based on mode
        if self.allocation_mode in ['prealloc', 'prealloc_fill_first']:
            # Smooth the initial distribution and use it as fixed allocation
            allocation_vector = self.smooth_distribution(b_distribution)
            with open(self.log_pth, "a") as log_file:
                log_file.write(f"\n[{self.get_allocation_method_name().upper()}]\n")
                log_file.write(f"Initial distribution: {[f'{p:.4f}' for p in b_distribution]}\n")
                log_file.write(f"Smoothed allocation vector: {[f'{p:.4f}' for p in allocation_vector]}\n")
                if self.allocation_mode == 'prealloc_fill_first':
                    log_file.write(f"Phase 1: Will fill each option once (scenarios 1-{len(b_domain)})\n")
                    log_file.write(f"Phase 2: Will use optimal allocation (scenarios {len(b_domain)+1}-{self.total_scenarios})\n")
                log_file.write("\n")
        else:
            # Online mode - allocation vector will be updated dynamically
            allocation_vector = b_distribution.copy()
        
        # Step 2: Generate scenarios and compute MIS estimates
        scenarios_data = []
        current_allocation = [0] * len(b_domain)  # Track n_k for each k
        
        for i in range(self.total_scenarios):
            # Decide which B option to use for this scenario (deterministic)
            if self.allocation_mode == 'prealloc_fill_first':
                option_index = self.should_allocate_to_option_fill_first(allocation_vector, current_allocation, i)
            else:
                option_index = self.should_allocate_to_option(allocation_vector, current_allocation)
            
            b_value = b_domain[option_index]
            b_fact = b_facts[option_index]
            current_allocation[option_index] += 1
            
            # Track which B values have been sampled
            b_values_sampled.add(option_index)
            
            # Check if this is the first time we have sampled all B values
            if s_full_index is None and len(b_values_sampled) == len(b_domain):
                s_full_index = i  # This is the scenario that completes the set
                with open(self.log_pth, "a") as log_file:
                    log_file.write(f"\n[MILESTONE] All B values have been sampled after scenario {i+1}\n")
            
            with open(self.log_pth, "a") as log_file:
                log_file.write(f"\n{'*'*60}\n")
                log_file.write(f"SCENARIO {i+1}/{self.total_scenarios}\n")
                
                # Add phase information for prealloc_fill_first
                if self.allocation_mode == 'prealloc_fill_first':
                    if i < len(b_domain):
                        log_file.write(f"PHASE 1 (Fill-first): Ensuring coverage\n")
                    else:
                        log_file.write(f"PHASE 2 (Optimal): Using allocation vector\n")
                
                log_file.write(f"Generating with B = {b_value} (option {option_index})\n")
                log_file.write(f"B Fact: {b_fact}\n")
                log_file.write(f"Current allocation (n_k): {dict(zip(b_domain, current_allocation))}\n")
                log_file.write(f"B values sampled so far: {sorted([b_domain[idx] for idx in b_values_sampled])}\n")
                log_file.write(f"{'*'*60}\n\n")
            
            # Generate scenario
            scenario = self.generate_scenario(a_facts, b_fact)
            
            with open(self.log_pth, "a") as log_file:
                log_file.write(f"\n[GENERATED SCENARIO]:\n{scenario}\n\n")
            
            # Compute likelihood ratios or probabilities based on mode
            if self.mode in ['verbal', 'verbal_small']:
                # Get P(X|A,B=b_k) / P(X|A) ratios for all k
                likelihood_ratios = []
                for k, b_fact_k in enumerate(b_facts):
                    ratio = self.get_likelihood_ratio_verbal_mode_aware(a_facts, scenario, b_fact_k)
                    likelihood_ratios.append(ratio)
                
                with open(self.log_pth, "a") as log_file:
                    log_file.write(f"\n[{self.mode.upper()} MODE - LIKELIHOOD RATIOS]\n")
                    for k, (b_val, ratio) in enumerate(zip(b_domain, likelihood_ratios)):
                        log_file.write(f"  P(X|A,B={b_val})/P(X|A) = {ratio:.4f}\n")
                
                # Store scenario data for verbal modes
                scenarios_data.append({
                    "scenario": scenario,
                    "b_value_used": b_value,
                    "b_index_used": option_index,
                    "likelihood_ratios": likelihood_ratios,
                    "p_b_given_scenario": None  # Will be filled later
                })
                
            else:  # string or string_scaled mode
                # Get actual probabilities
                prob_x_given_a, prob_x_given_a_and_bk = self.get_probabilities_string(a_facts, scenario, b_facts)
                
                # Calculate likelihood ratios for logging purposes
                likelihood_ratios = []
                for prob_ab in prob_x_given_a_and_bk:
                    if prob_x_given_a > 0:
                        likelihood_ratios.append(prob_ab / prob_x_given_a)
                    else:
                        likelihood_ratios.append(0.0)
                
                # Apply scaling for string_scaled mode
                if self.mode == 'string_scaled' and len(likelihood_ratios) > 0:
                    # Check if all ratios are below 1 or all above 1
                    all_below_1 = all(ratio < 1.0 for ratio in likelihood_ratios if ratio > 0)
                    all_above_1 = all(ratio > 1.0 for ratio in likelihood_ratios if ratio > 0)
                    
                    if all_below_1:
                        # Scale up so the maximum ratio becomes 1
                        max_ratio = max(likelihood_ratios)
                        if max_ratio > 0:
                            scale_factor = 1.0 / max_ratio
                            likelihood_ratios = [ratio * scale_factor for ratio in likelihood_ratios]
                            
                            with open(self.log_pth, "a") as log_file:
                                log_file.write(f"[SCALING] All ratios below 1, scaled up by factor {scale_factor:.4f}\n")
                    
                    elif all_above_1:
                        # Scale down so the minimum ratio becomes 1
                        min_ratio = min(likelihood_ratios)
                        if min_ratio > 0:
                            scale_factor = 1.0 / min_ratio
                            likelihood_ratios = [ratio * scale_factor for ratio in likelihood_ratios]
                            
                            with open(self.log_pth, "a") as log_file:
                                log_file.write(f"[SCALING] All ratios above 1, scaled down by factor {scale_factor:.4f}\n")
                
                with open(self.log_pth, "a") as log_file:
                    log_file.write(f"\n[{self.mode.upper()} MODE - PROBABILITIES]\n")
                    log_file.write(f"  P(X|A) = {prob_x_given_a:.6f}\n")
                    for k, (b_val, prob, ratio) in enumerate(zip(b_domain, prob_x_given_a_and_bk, likelihood_ratios)):
                        log_file.write(f"  P(X|A,B={b_val}) = {prob:.6f} (ratio: {ratio:.4f})\n")
                
                # Store scenario data for string modes with actual probabilities
                scenarios_data.append({
                    "scenario": scenario,
                    "b_value_used": b_value,
                    "b_index_used": option_index,
                    "prob_x_given_a": prob_x_given_a,  # Store for string modes
                    "prob_x_given_a_and_bk": prob_x_given_a_and_bk,  # Store for string modes
                    "likelihood_ratios": likelihood_ratios,  # Keep for logging (possibly scaled)
                    "p_b_given_scenario": None  # Will be filled later
                })
            
            # Get P(B|X,A) from OpenAI
            p_b_given_scenario = self.estimate_b_given_scenario(a_facts, scenario, b_question, b_domain)
            
            # Update the scenario data with P(B|X,A)
            scenarios_data[-1]["p_b_given_scenario"] = p_b_given_scenario
            
            # Recalculate weights for all scenarios
            weights = []
            for j, data in enumerate(scenarios_data):
                if self.mode in ['verbal', 'verbal_small']:
                    # Verbal modes: weight = 1 / Σₖ(nₖ * ratio_k)
                    denominator = 0.0
                    for k in range(len(b_domain)):
                        denominator += current_allocation[k] * data["likelihood_ratios"][k]
                    
                    if denominator > 0:
                        weight = 1.0 / denominator
                    else:
                        weight = 0.0
                        
                else:  # string or string_scaled mode
                    # String modes: weight = P(X|A) / Σₖ(nₖ * P(X|A,B=bₖ))
                    numerator = data["prob_x_given_a"]
                    denominator = 0.0
                    for k in range(len(b_domain)):
                        denominator += current_allocation[k] * data["prob_x_given_a_and_bk"][k]
                    
                    if denominator > 0:
                        weight = numerator / denominator
                    else:
                        weight = 0.0
                
                weights.append(weight)
            
            # Normalize weights
            total_weight = sum(weights)
            if total_weight > 0:
                normalized_weights = [w / total_weight for w in weights]
            else:
                normalized_weights = [0.0] * len(weights)
            
            # Track the unnormalized total weight
            weight_values.append(total_weight)
            
            # Compute ESS from normalized weights
            current_ess = self.compute_effective_sample_size(normalized_weights)
            ess_values.append(current_ess)
            
            # Write weight value to file (first scenario has no comma, others do)
            with open(self.weights_pth, "a") as weights_file:
                if i == 0:
                    weights_file.write(f"{total_weight:.6f}")
                else:
                    # Use semicolon if this is s_full, comma otherwise
                    separator = "; " if s_full_index == i else ", "
                    weights_file.write(f"{separator}{total_weight:.6f}")
            
            # Write ESS value to file
            with open(self.ess_pth, "a") as ess_file:
                if i == 0:
                    ess_file.write(f"{current_ess:.6f}")
                else:
                    # Use semicolon if this is s_full, comma otherwise
                    separator = "; " if s_full_index == i else ", "
                    ess_file.write(f"{separator}{current_ess:.6f}")
            
            # Log the unnormalized total weight and ESS
            with open(self.log_pth, "a") as log_file:
                log_file.write(f"\n[WEIGHT CALCULATION - After Scenario {i+1}]\n")
                log_file.write(f"  Mode: {self.mode}\n")
                log_file.write(f"  Unnormalized total weight: {total_weight:.6f}\n")
                log_file.write(f"  Individual weights (unnormalized): {[f'{w:.6f}' for w in weights]}\n")
                log_file.write(f"  Normalized weights: {[f'{w:.6f}' for w in normalized_weights]}\n")
                log_file.write(f"  Effective Sample Size (ESS): {current_ess:.6f}\n")
            
            # Compute current estimates using normalized weights
            current_estimates = [0.0] * len(b_domain)
            for j, (data, norm_weight) in enumerate(zip(scenarios_data, normalized_weights)):
                for b_idx in range(len(b_domain)):
                    current_estimates[b_idx] += norm_weight * data["p_b_given_scenario"][b_idx]
            
            # Update allocation vector for next iteration if in online mode
            if self.allocation_mode == 'online':
                allocation_vector = current_estimates.copy()
            
            # Compute KL divergences after this scenario if ground truth is provided
            if ground_truth is not None:
                try:
                    scenario_kl, scenario_kl_unweighted, scenario_kl_flattened = self.compute_kl_divergence_variants(current_estimates, ground_truth)
                    kl_values.append(scenario_kl)
                    kl_unweighted_values.append(scenario_kl_unweighted)
                    kl_flattened_values.append(scenario_kl_flattened)
                    
                    # Write KL values to files with appropriate separator
                    separator = "; " if s_full_index == i else ", "
                    
                    with open(self.kl_pth, "a") as kl_file:
                        kl_file.write(f"{separator}{scenario_kl:.6f}")
                    
                    with open(self.kl_unweighted_pth, "a") as kl_file:
                        kl_file.write(f"{separator}{scenario_kl_unweighted:.6f}")
                    
                    with open(self.kl_flattened_pth, "a") as kl_file:
                        kl_file.write(f"{separator}{scenario_kl_flattened:.6f}")
                        
                except ValueError as e:
                    with open(self.log_pth, "a") as log_file:
                        log_file.write(f"[ERROR] Could not compute KL after scenario {i+1}: {e}\n")
            
            with open(self.log_pth, "a") as log_file:
                log_file.write(f"\n[SCENARIO {i+1} RESULTS]\n")
                log_file.write(f"  P(B|scenario):\n")
                for k, (option, prob) in enumerate(zip(b_domain, p_b_given_scenario)):
                    log_file.write(f"    P(B={option}|scenario) = {prob:.4f}\n")
                log_file.write(f"  Current MIS estimates:\n")
                for k, (option, est) in enumerate(zip(b_domain, current_estimates)):
                    log_file.write(f"    P(B={option}|A) = {est:.4f} (initial: {b_distribution[k]:.4f})")
                    if ground_truth is not None:
                        log_file.write(f" (true: {ground_truth[k]:.4f})")
                    log_file.write("\n")
                if ground_truth is not None and len(kl_values) > 1:
                    log_file.write(f"  KL divergence after scenario {i+1}: {kl_values[-1]:.6f}\n")
                log_file.write(f"  ESS after scenario {i+1}: {current_ess:.6f}\n")
        
        # Step 3: Final estimates are the last computed current_estimates
        final_estimates = current_estimates
        
        with open(self.log_pth, "a") as log_file:
            log_file.write(f"\n{'='*60}\n")
            log_file.write(f"FINAL RESULTS\n")
            log_file.write(f"{'='*60}\n")
            log_file.write(f"Initial estimates vs Final MIS estimates:\n")
            for i, (option, init_prob, final_prob) in enumerate(zip(b_domain, b_distribution, final_estimates)):
                diff = final_prob - init_prob
                log_file.write(f"  P(B={option}|A): {init_prob:.4f} -> {final_prob:.4f} ({diff:+.4f})")
                if ground_truth is not None:
                    log_file.write(f" (true: {ground_truth[i]:.4f})")
                log_file.write("\n")
            log_file.write(f"Final scenario allocation (n_k): {dict(zip(b_domain, current_allocation))}\n")
            log_file.write(f"Allocation method: {self.get_allocation_method_name()}\n")
            if self.allocation_mode == 'prealloc_fill_first':
                log_file.write(f"Fill-first phase completed after scenario {len(b_domain)}\n")
            log_file.write(f"Final unnormalized total weight: {total_weight:.6f}\n")
            log_file.write(f"Final ESS: {current_ess:.6f}\n")
            if s_full_index is not None:
                log_file.write(f"All B values were first sampled after scenario {s_full_index + 1}\n")
            else:
                log_file.write(f"WARNING: Not all B values were sampled in {self.total_scenarios} scenarios\n")
            if ground_truth is not None and len(kl_values) > 0:
                final_kl = kl_values[-1]
                final_kl_unweighted = kl_unweighted_values[-1]
                final_kl_flattened = kl_flattened_values[-1]
                
                log_file.write(f"Final KL divergences:\n")
                log_file.write(f"  Normal KL: {final_kl:.6f} (improvement: {kl_values[0] - final_kl:+.6f})\n")
                log_file.write(f"  Unweighted KL: {final_kl_unweighted:.6f} (improvement: {kl_unweighted_values[0] - final_kl_unweighted:+.6f})\n")
                log_file.write(f"  Flattened KL: {final_kl_flattened:.6f} (improvement: {kl_flattened_values[0] - final_kl_flattened:+.6f})\n")
            log_file.write(f"{'='*60}\n\n")
        
        # Write newlines to complete the KL output files
        if ground_truth is not None:
            with open(self.kl_pth, "a") as kl_file:
                kl_file.write("\n")
            
            with open(self.kl_unweighted_pth, "a") as kl_file:
                kl_file.write("\n")
            
            with open(self.kl_flattened_pth, "a") as kl_file:
                kl_file.write("\n")
        
        # Write newline to complete the weights line for this input
        with open(self.weights_pth, "a") as weights_file:
            weights_file.write("\n")
        
        # Write newline to complete the ESS line for this input
        with open(self.ess_pth, "a") as ess_file:
            ess_file.write("\n")
        
        return dict(zip(b_domain, final_estimates))

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
    """Main function to run maieutic prompting with MIS on input file."""

    #############################################################################################
    # Define folder paths and file names
    input_folder = "/home/rliu79/reasoning/experiment_maieutic_compare" 
    output_folder = "/home/rliu79/reasoning/experiment_maieutic_compare/verbal_openai_output"
    # IMPORTANT: Replace with your actual OpenAI API key
    OPENAI_API_KEY = 'Your_OpenAI_API_Key_Here'
    #############################################################################################

    # Define file paths
    config_path = os.path.join(input_folder, "maieutic_config_verbal_openai.json")
    input_path = os.path.join(input_folder, "input.txt")
    output_path = os.path.join(output_folder, "output.txt")
    log_path = os.path.join(output_folder, "log.txt")
    kl_path = os.path.join(output_folder, "kl_output.txt")
    weights_path = os.path.join(output_folder, "weights_output.txt")
    ess_path = os.path.join(output_folder, "ESS_output.txt")


    client = OpenAI(api_key=OPENAI_API_KEY)

    # Initialize maieutic system
    maieutic = MaieuticPromptingMIS(
        config_path=config_path,
        log_path=log_path,
        kl_path=kl_path,
        model_id="gpt-4o-2024-08-06",
        client=client,
        weights_path=weights_path,
        ess_path=ess_path
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
    print(f"Unweighted KL tracking available at {maieutic.kl_unweighted_pth}")
    print(f"Flattened KL tracking available at {maieutic.kl_flattened_pth}")
    print(f"Weights tracking available at {weights_path}")
    print(f"ESS tracking available at {ess_path}")


if __name__ == "__main__":
    main()