##############################################################################################################
# Maieutic Prompting Agentic Workflow - Revised with Fully Deterministic Mode
#
# In deterministic mode, we simplify the importance weight calculation:
# w(x) = P(X = x | A = a) / P(X = x | B = b, A = a)
# where b is the specific value used to generate scenario x
#
# Modes:
# - "normal": Original implementation using verbal likelihood ratios (OpenAI only)
# - "fully_deterministic": Use small model (Llama3) to compute exact token probabilities
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
        
        # Get mode from config
        self.mode = self.config['model_settings'].get('mode', 'normal')
        
        # Initialize small model if in fully_deterministic, partial_deterministic, or minimal mode
        self.small_model = None
        self.small_tokenizer = None
        if self.mode in ['fully_deterministic', 'partial_deterministic', 'minimal']:
            self._initialize_small_model()
        
        # Initialize log file with header
        with open(self.log_pth, "w") as log_file:
            log_file.write(f"=== Maieutic Prompting Log - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===\n")
            log_file.write(f"Mode: {self.mode}\n\n")
        
        # Initialize KL divergence output file
        with open(self.kl_pth, "w") as kl_file:
            pass  # Create empty file, no headers or explanations
    
    def _initialize_small_model(self):
        """Initialize the small model (Llama3) for token probability computation."""
        try:
            model_name = self.config['model_settings']['small_model']['model_name']
            
            with open(self.log_pth, "a") as log_file:
                log_file.write(f"[INIT] Loading small model: {model_name}\n")
            
            # Load tokenizer and model
            self.small_tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            # Check if GPU is available
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            if device.type != "cuda":
                with open(self.log_pth, "a") as log_file:
                    log_file.write("[WARN] GPU not available, using CPU. This will be slower.\n")
            
            # Load model with appropriate settings
            self.small_model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
                device_map="auto" if device.type == "cuda" else None
            )
            
            if device.type != "cuda":
                self.small_model = self.small_model.to(device)
            
            self.small_model.eval()  # Set to evaluation mode
            
            # Set padding token if not set
            if self.small_tokenizer.pad_token is None:
                self.small_tokenizer.pad_token = self.small_tokenizer.eos_token
            
            with open(self.log_pth, "a") as log_file:
                log_file.write(f"[INIT] Small model loaded successfully on {device}\n\n")
                
        except Exception as e:
            with open(self.log_pth, "a") as log_file:
                log_file.write(f"[ERROR] Failed to initialize small model: {e}\n")
            raise RuntimeError(f"Failed to initialize small model: {e}")
    
    def get_token_log_probability(self, prompt: str, completion: str) -> float:
        """
        Calculate the log probability of generating the completion given the prompt using the small model.
        Returns the sum of log probabilities for all tokens in the completion.
        """
        if self.small_model is None:
            raise RuntimeError("Small model not initialized")
        
        try:
            # Combine prompt and completion
            full_text = prompt + completion
            
            # Tokenize
            prompt_encoding = self.small_tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
            full_encoding = self.small_tokenizer(full_text, return_tensors="pt", add_special_tokens=True)
            
            # Move to same device as model
            device = next(self.small_model.parameters()).device
            prompt_ids = prompt_encoding.input_ids.to(device)
            full_ids = full_encoding.input_ids.to(device)
            
            # Get the number of tokens in the prompt
            prompt_length = prompt_ids.shape[1]
            
            # Forward pass
            with torch.no_grad():
                outputs = self.small_model(full_ids)
                logits = outputs.logits
            
            # Calculate log probabilities
            # Shift logits and labels for next-token prediction
            shift_logits = logits[..., prompt_length-1:-1, :].contiguous()
            shift_labels = full_ids[..., prompt_length:].contiguous()
            
            # Calculate log probabilities for each token
            log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)
            
            # Gather the log probabilities of the actual tokens
            token_log_probs = log_probs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)
            
            # Sum log probabilities
            total_log_prob = token_log_probs.sum().item()
            
            return total_log_prob
            
        except Exception as e:
            with open(self.log_pth, "a") as log_file:
                log_file.write(f"[ERROR] Failed to compute token log probability: {e}\n")
            raise RuntimeError(f"Failed to compute token log probability: {e}")
    
    def single_LLM_call(self, json_prompt, client, name, replacements={}, additional_message=None, return_logprobs=False):
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
        kwargs = {
            "model": self.model_ID,
            "messages": messages,
            "temperature": self.temperature
        }
        
        if return_logprobs:
            kwargs["logprobs"] = True
        
        response = client.chat.completions.create(**kwargs)
        
        response_content = response.choices[0].message.content
        
        with open(self.log_pth, "a") as log_file:
            log_file.write(f"[RESPONSE]: {response_content}\n")
            log_file.write("-" * 80 + "\n\n")
        
        self.LLM_budget_left -= 1

        # Return the actual generation
        if return_logprobs:
            return response_content, response
        else:
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
    
    def extract_logprobs_from_response(self, response) -> float:
        """Extract and sum log probabilities from OpenAI response."""
        try:
            total_logprob = 0.0
            if response.choices[0].logprobs and response.choices[0].logprobs.content:
                for token_info in response.choices[0].logprobs.content:
                    if token_info.logprob is not None:
                        total_logprob += token_info.logprob
            return total_logprob
        except Exception as e:
            with open(self.log_pth, "a") as log_file:
                log_file.write(f"[ERROR] Failed to extract logprobs: {e}\n")
            raise RuntimeError(f"Failed to extract logprobs: {e}")
    
    def generate_scenario_with_small_model(self, a_facts: List[str], b_fact: str) -> Tuple[str, float]:
        """Generate scenario using Llama3 and return both scenario and its log probability."""
        if self.small_model is None:
            raise RuntimeError("Small model not initialized")
        
        try:
            # Build the prompt using the scenario_generation template
            scenario_prompt = self.config['prompts']['scenario_generation']
            prompt = ""
            
            for role, content in scenario_prompt:
                # Replace placeholders
                content_filled = content.replace("{A}", self.concatenate_facts(a_facts))
                content_filled = content_filled.replace("{B}", b_fact)
                
                if role == "system":
                    prompt += f"System: {content_filled}\n"
                elif role == "user":
                    prompt += f"User: {content_filled}\nAssistant: "
                elif role == "assistant":
                    prompt += f"{content_filled}\n"
            
            # Tokenize the prompt
            inputs = self.small_tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
            device = next(self.small_model.parameters()).device
            input_ids = inputs.input_ids.to(device)
            attention_mask = inputs.attention_mask.to(device)
            
            # Generate with temperature
            with torch.no_grad():
                outputs = self.small_model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=200,
                    temperature=self.config['model_settings']['scenario_generation_temperature'],
                    do_sample=True,
                    pad_token_id=self.small_tokenizer.pad_token_id,
                    return_dict_in_generate=True,
                    output_scores=True
                )
            
            # Extract generated tokens (excluding prompt)
            generated_ids = outputs.sequences[0][input_ids.shape[1]:]
            generated_text = self.small_tokenizer.decode(generated_ids, skip_special_tokens=True)
            
            # Calculate log probability of the generated sequence
            # Get the scores for each generated token
            scores = torch.stack(outputs.scores, dim=0)  # Shape: [seq_len, vocab_size]
            
            # Convert to log probabilities
            log_probs = torch.nn.functional.log_softmax(scores, dim=-1)
            
            # Get the log prob of each generated token
            total_log_prob = 0.0
            for i, token_id in enumerate(generated_ids):
                if i < len(log_probs):
                    total_log_prob += log_probs[i, token_id].item()
            
            with open(self.log_pth, "a") as log_file:
                log_file.write(f"\n[LLAMA3 SCENARIO GENERATION]\n")
                log_file.write(f"Prompt: {prompt[:200]}...\n")
                log_file.write(f"Generated: {generated_text}\n")
                log_file.write(f"Log P(X|A,B): {total_log_prob:.4f}\n\n")
            
            return generated_text.strip(), total_log_prob
            
        except Exception as e:
            with open(self.log_pth, "a") as log_file:
                log_file.write(f"[ERROR] Failed to generate scenario with small model: {e}\n")
            raise RuntimeError(f"Failed to generate scenario with small model: {e}")
    
    def generate_scenario(self, a_facts: List[str], b_fact: str, return_logprobs: bool = False) -> Any:
        """Generate a scenario X=x given A=a and B=b (as facts)."""
        # Use small model for minimal mode
        if self.mode == 'minimal':
            scenario, log_prob = self.generate_scenario_with_small_model(a_facts, b_fact)
            if return_logprobs:
                return scenario, log_prob
            else:
                return scenario
        
        # Original OpenAI logic for other modes
        # Use higher temperature for scenario generation
        self.temperature = self.config['model_settings']['scenario_generation_temperature']
        
        # Concatenate facts properly
        a_fact_string = self.concatenate_facts(a_facts)
        
        result = self.single_LLM_call(
            self.config['prompts']['scenario_generation'],
            self.client,
            "scenario_generation",
            {
                "A": a_fact_string,
                "B": b_fact
            },
            return_logprobs=return_logprobs
        )
        
        # Reset temperature
        self.temperature = self.config['model_settings']['evaluation_temperature']
        
        if return_logprobs:
            response_content, response = result
            total_logprob = self.extract_logprobs_from_response(response)
            return response_content.strip(), total_logprob
        else:
            return result.strip()
    
    def compute_importance_weight_with_probabilities(self, a_facts: List[str], x_scenario: str, 
                                                   b_fact: str, log_prob_x_given_a_and_b: Optional[float] = None) -> float:
        """
        Compute importance weight w(x) for deterministic modes.
        w(x) = P(X=x|A=a) / P(X=x|A=a,B=b)
        """
        if self.mode in ['fully_deterministic', 'partial_deterministic', 'minimal']:
            # Get P(X|A,B)
            if log_prob_x_given_a_and_b is not None:
                # Use provided logprob from OpenAI (partial_deterministic mode)
                pass
            else:
                # Compute using small model (fully_deterministic mode)
                # Build the prompt for P(X|A,B)
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
                
                log_prob_x_given_a_and_b = self.get_token_log_probability(prompt_a_and_b, x_scenario)
            
            # Get P(X|A) using small model (same for both partial and fully deterministic)
            scenario_prompt_no_b = self.config['prompts']['scenario_generation_no_b']
            messages_a_only = []
            for role, content in scenario_prompt_no_b:
                content_filled = content.replace("{A}", self.concatenate_facts(a_facts))
                messages_a_only.append({"role": role, "content": content_filled})
            
            prompt_a_only = ""
            for msg in messages_a_only:
                if msg["role"] == "system":
                    prompt_a_only += f"System: {msg['content']}\n"
                elif msg["role"] == "user":
                    prompt_a_only += f"User: {msg['content']}\nAssistant: "
                elif msg["role"] == "assistant":
                    prompt_a_only += f"{msg['content']}\n"
            
            log_prob_x_given_a = self.get_token_log_probability(prompt_a_only, x_scenario)
            
            # w(x) = exp(log P(X=x|A=a) - log P(X=x|A=a,B=b))
            log_weight = log_prob_x_given_a - log_prob_x_given_a_and_b
            
            # Prevent overflow/underflow
            log_weight = max(min(log_weight, 10), -10)
            weight = math.exp(log_weight)
            
            with open(self.log_pth, "a") as log_file:
                log_file.write(f"[DETERMINISTIC WEIGHT COMPUTATION]\n")
                if self.mode == 'partial_deterministic':
                    log_file.write(f"  log P(X|A) = {log_prob_x_given_a:.4f} (from Llama3)\n")
                    log_file.write(f"  log P(X|A,B) = {log_prob_x_given_a_and_b:.4f} (from OpenAI)\n")
                elif self.mode == 'minimal':
                    log_file.write(f"  log P(X|A) = {log_prob_x_given_a:.4f} (from Llama3)\n")
                    log_file.write(f"  log P(X|A,B) = {log_prob_x_given_a_and_b:.4f} (from Llama3 generation)\n")
                else:
                    log_file.write(f"  log P(X|A) = {log_prob_x_given_a:.4f}\n")
                    log_file.write(f"  log P(X|A,B) = {log_prob_x_given_a_and_b:.4f}\n")
                log_file.write(f"  log w(x) = {log_weight:.4f}\n")
                log_file.write(f"  w(x) = {weight:.4f}\n\n")
            
            return weight
        else:
            # This should not be called in normal mode
            raise RuntimeError("compute_importance_weight_with_probabilities called in normal mode")
    
    def compute_importance_weight(self, a_facts: List[str], x_scenario: str, b_facts: List[str], 
                                  b_probs: List[float]) -> float:
        """Compute importance weight w(x) for a scenario."""
        if self.mode == 'fully_deterministic':
            # In deterministic mode, we know which specific b was used to generate x
            # Find which b_fact was used (this should be passed or tracked, but for now we'll use the first matching one)
            # In actual implementation, we should track which b was used for each scenario
            # For now, we'll compute the weight using the deterministic formula
            # Note: This is a limitation - we need to know which b was used
            with open(self.log_pth, "a") as log_file:
                log_file.write(f"[WARN] In deterministic mode, need to track which b was used for scenario generation\n")
            
            # Use the first b_fact as a placeholder - this should be fixed by tracking b allocation
            return self.compute_importance_weight_deterministic(a_facts, x_scenario, b_facts[0])
        else:
            # Original implementation for normal mode
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
            log_file.write(f"Mode: {self.mode}\n")
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
            # Decide which B option to use for this scenario (deterministic)
            option_index = self.should_allocate_to_option(b_distribution, current_allocation)
            b_value = b_domain[option_index]
            b_fact = b_facts[option_index]
            current_allocation[option_index] += 1
            
            with open(self.log_pth, "a") as log_file:
                log_file.write(f"\n{'*'*60}\n")
                log_file.write(f"SCENARIO {i+1}/{self.total_scenarios}\n")
                log_file.write(f"Generating with B = {b_value} (deterministically chosen)\n")
                log_file.write(f"B Fact: {b_fact}\n")
                log_file.write(f"Current allocation: {dict(zip(b_domain, current_allocation))}\n")
                log_file.write(f"{'*'*60}\n\n")
            
            # Generate scenario
            if self.mode in ['partial_deterministic', 'minimal']:
                scenario, log_prob_openai = self.generate_scenario(a_facts, b_fact, return_logprobs=True)
            else:
                scenario = self.generate_scenario(a_facts, b_fact)
            
            with open(self.log_pth, "a") as log_file:
                log_file.write(f"\n[GENERATED SCENARIO]:\n{scenario}\n\n")
            
            # Compute importance weight
            if self.mode == 'partial_deterministic':
                # Use OpenAI logprob for P(X|A,B) and Llama3 for P(X|A)
                weight = self.compute_importance_weight_with_probabilities(
                    a_facts, scenario, b_fact, log_prob_x_given_a_and_b=log_prob_openai
                )
            elif self.mode == 'minimal':
                # Use Llama3 logprob from generation for P(X|A,B) and compute P(X|A)
                weight = self.compute_importance_weight_with_probabilities(
                    a_facts, scenario, b_fact, log_prob_x_given_a_and_b=log_prob_openai
                )
            elif self.mode == 'fully_deterministic':
                # Use Llama3 for both probabilities
                weight = self.compute_importance_weight_with_probabilities(a_facts, scenario, b_fact)
            else:
                # In normal mode, use the original method
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
    OPENAI_API_KEY = 'YOUR_OPENAI_API_KEY'
    
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