import json
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple
import os
import openai
from loguru import logger
from dotenv import load_dotenv
import re
from pathlib import Path

load_dotenv()

# Define the system prompt for Tree of Thoughts with placeholders
TREE_OF_THOUGHTS_TEMPLATE = """
You are an expert problem-solving agent designed to not only solve complex problems but also critically evaluate the quality of your thought process and final answers. 
Your task is to follow a structured approach to generate solutions, assess your thoughts, and provide a rating for each on a scale of 0.1 to 1.0. 
This rating should reflect the accuracy and quality of your reasoning and final answer.

### Instructions:

1. **Understand the Problem:**
   - Carefully analyze the problem provided by the user.
   - Break down the problem into smaller, manageable parts if necessary.
   - Formulate a clear understanding of the problem before proceeding.

2. **Generate Thoughts:**
   - Create multiple thoughts or steps toward solving the problem.
   - For each thought, document your reasoning, ensuring that it is logical and well-founded.

3. **Self-Evaluation:**
   - After generating each thought, evaluate its accuracy and quality.
   - Assign an evaluation score between 0.1 and 1.0. Use the following guidelines:
     - **0.1 to 0.4:** The thought is flawed, inaccurate, or incomplete.
     - **0.5 to 0.7:** The thought is partially correct but may lack detail or full accuracy.
     - **0.8 to 1.0:** The thought is accurate, complete, and well-reasoned.

4. **Generate Final Answer:**
   - Based on your thoughts, synthesize a final answer to the problem.
   - Ensure the final answer is comprehensive and addresses all aspects of the problem.

5. **Final Evaluation:**
   - Evaluate the overall quality and accuracy of your final answer.
   - Provide a final evaluation score based on the same 0.1 to 1.0 scale.
   
### Problem Context:
Domain: {domain}
Question: {question}
"""


def string_to_dict(thought_string):
    """
    Convert a thought string to a dictionary.
    """
    return eval(thought_string)


class Thought:
    """
    Represents a thought with its evaluation.
    """
    def __init__(self, thought: str, evaluation: Optional[float] = None):
        """
        Initialize a Thought object.
        
        Args:
            thought (str): The content of the thought.
            evaluation (Optional[float]): The evaluation score (0.1 to 1.0).
        """
        self.thought = thought
        self.evaluation = evaluation
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "thought": self.thought,
            "evaluation": self.evaluation
        }

    def __str__(self) -> str:
        return f"Thought(thought={self.thought[:50]}..., evaluation={self.evaluation})"


class ToTAgent:
    """
    Tree of Thoughts agent that performs reasoning and evaluation.
    """
    def __init__(
        self,
        api_key: str,
        model_id: str = "gpt-3.5-turbo",
        temperature: float = 0.0,
        max_tokens: int = 3000,
        id: str = None
    ):
        """
        Initialize the ToT agent.

        Args:
            api_key (str): OpenAI API key
            model_id (str): Model to use for completions
            temperature (float): Sampling temperature
            max_tokens (int): Maximum tokens for response
            id (str): Unique identifier for the agent
        """
        self.id = id or uuid.uuid4().hex
        # Set up OpenAI client - handle both old and new API versions
        self.api_key = api_key
        openai.api_key = api_key  # For older OpenAI versions
        try:
            # Try new-style client initialization (v1.0+)
            self.client = openai.OpenAI(api_key=api_key)
            self.new_api = True
        except (AttributeError, TypeError):
            # Fall back to older API version
            self.client = openai
            self.new_api = False
            
        self.model_id = model_id
        self.temperature = temperature
        self.max_tokens = max_tokens
    
    def run(self, task: str, domain: str = "", parent_agent=None) -> Dict[str, Any]:
        """
        Run a single thought generation and evaluation task.
        
        Args:
            task (str): The task or question to process
            domain (str): Optional domain context
            parent_agent: Parent agent to track budget
            
        Returns:
            Dict[str, Any]: The output containing thought and evaluation
        """
        system_prompt = TREE_OF_THOUGHTS_TEMPLATE.format(question=task, domain=domain)
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "Think step by step to solve this problem."}
        ]
        
        if self.new_api:
            # New OpenAI API (v1.0+)
            response = self.client.chat.completions.create(
                model=self.model_id,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            response_text = response.choices[0].message.content
        else:
            # Older OpenAI API
            response = self.client.ChatCompletion.create(
                model=self.model_id,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            response_text = response['choices'][0]['message']['content']
        
        # Decrement budget in parent agent if provided
        if parent_agent is not None:
            parent_agent.LLM_budget_left -= 4
            logger.info(f"Budget decreased to {parent_agent.LLM_budget_left}")
        
        # Parse the response to extract thought and evaluation
        evaluation = None
        for line in response_text.split('\n'):
            if "evaluation:" in line.lower() or "score:" in line.lower():
                # Extract numerical evaluation
                numbers = re.findall(r"[-+]?\d*\.\d+|\d+", line)
                if numbers:
                    evaluation = float(numbers[0])
                    break
        
        # If no explicit evaluation found, set a default middle value
        if evaluation is None:
            evaluation = 0.5
            
        return {
            "thought": response_text,
            "evaluation": evaluation
        }
    
    def derive_distribution(self, thought: str, domain: str, question: str, parent_agent=None) -> List[float]:
        """
        Given a thought, derive a probability distribution over the domain options.
        # CHANGED: Now counts towards the LLM budget.
        
        Args:
            thought (str): The thought to analyze
            domain (str): The domain of possible answers
            question (str): The original question
            parent_agent: Parent agent to track budget  # CHANGED: Added parent_agent parameter
            
        Returns:
            List[float]: Probability distribution over domain options
        """
        try:
            # Parse domain string to extract options
            domain_options = self._parse_domain(domain)
            num_options = len(domain_options)
            
            # For safety, limit thought length
            #truncated_thought = thought[:1000] + "..." if len(thought) > 1000 else thought
            truncated_thought = thought
            # Create a prompt to derive a distribution
            prompt = f"""
Based on the following thought process, give a probability distribution over these options: {domain_options}.

Question: {question}

Thought process:
{truncated_thought}

Your response MUST be ONLY a list of probabilities that sum to 1.0, like: [0.7, 0.3]
Do not include any explanations or other text. Only return the list of probabilities.
"""
            
            messages = [
                {"role": "system", "content": "You must only output a probability distribution as a list of numbers in brackets, nothing else."},
                {"role": "user", "content": prompt}
            ]
            
            logger.info(f"Deriving distribution for domain options: {domain_options}")
            
            if self.new_api:
                # New OpenAI API (v1.0+)
                response = self.client.chat.completions.create(
                    model=self.model_id,
                    messages=messages,
                    temperature=0.0,  # Use 0 temperature for consistency
                    max_tokens=50  # Keep very short to avoid unnecessary text
                )
                response_text = response.choices[0].message.content
            else:
                # Older OpenAI API
                response = self.client.ChatCompletion.create(
                    model=self.model_id,
                    messages=messages,
                    temperature=0.0,
                    max_tokens=50
                )
                response_text = response['choices'][0]['message']['content']
            
            # CHANGED: Now distribution calls count towards the budget
            if parent_agent is not None:
                parent_agent.LLM_budget_left -= 4
                logger.info(f"Budget decreased to {parent_agent.LLM_budget_left} (distribution call)")
            
            logger.info(f"Raw distribution response: {response_text}")
            
            # Parse the distribution
            distribution = self._parse_distribution(response_text, num_options)
            logger.info(f"Parsed distribution: {distribution}")
            return distribution
            
        except Exception as e:
            logger.error(f"Error deriving distribution: {str(e)}")
            # Return a uniform distribution as fallback
            domain_options = self._parse_domain(domain)
            return [1.0 / len(domain_options)] * len(domain_options)
    
    def _parse_domain(self, domain: str) -> List[str]:
        """Parse the domain string to extract options."""
        # Try to match a list-like structure
        match = re.search(r'\[(.*)\]', domain)
        if match:
            # Split by comma and clean up
            options = [opt.strip() for opt in match.group(1).split(',')]
            return options
        else:
            # If no list-like structure, split by comma or space
            if ',' in domain:
                return [opt.strip() for opt in domain.split(',')]
            else:
                return [opt.strip() for opt in domain.split()]
    
    def _parse_distribution(self, response_text: str, num_options: int) -> List[float]:
        """Parse the distribution from the response text."""
        # Clean up the text - remove any non-essential characters
        clean_text = response_text.strip()
        logger.info(f"Cleaning text for distribution: {clean_text}")
        
        # Try to extract array of numbers
        match = re.search(r'\[(.*?)\]', clean_text)
        if match:
            try:
                # Get just the content inside the brackets
                bracket_content = match.group(1).strip()
                logger.info(f"Found bracketed content: {bracket_content}")
                
                # Split by comma
                number_strings = [x.strip() for x in bracket_content.split(',')]
                
                # Convert to floats
                if len(number_strings) == num_options:
                    probs = [float(x) for x in number_strings]
                    total = sum(probs)
                    
                    # Normalize to ensure they sum to 1
                    if total > 0:
                        logger.info(f"Parsed distribution: {probs}, sum={total}")
                        normalized = [p/total for p in probs]
                        return normalized
            except Exception as e:
                logger.error(f"Error parsing bracketed content: {str(e)}")
                
        # If we couldn't parse it from brackets, try finding numbers
        logger.info("Bracket parsing failed, extracting numbers directly")
        numbers = re.findall(r"[-+]?\d*\.\d+|\d+", clean_text)
        logger.info(f"Found numbers: {numbers}")
        
        if numbers and len(numbers) >= num_options:
            try:
                # Take only the required number of values
                probs = [float(n) for n in numbers[:num_options]]
                total = sum(probs)
                
                # Normalize
                if total > 0:
                    logger.info(f"Using direct number extraction: {probs}, sum={total}")
                    return [p/total for p in probs]
            except Exception as e:
                logger.error(f"Error parsing direct numbers: {str(e)}")
        
        # Ultimate fallback - return uniform distribution
        logger.warning(f"Failed to parse any distribution, using uniform fallback for {num_options} options")
        return [1.0 / num_options] * num_options


class ToTBFSAgent:
    """
    A class to perform Breadth-First Search (BFS) using the Tree of Thoughts approach.
    
    This approach follows the ToT-BFS algorithm from the original code.
    """

    def __init__(
        self,
        api_key: str = None,
        model_id: str = "gpt-3.5-turbo",
        temperature: float = 0.0,
        max_tokens: int = 3000,
        max_loops: int = 5,
        breadth_limit: int = 3,
        number_of_agents: int = 3,
        log_path: str = None,
        id: str = None,
        LLM_budget: int = 30,
    ):
        """
        Initialize the ToTBFSAgent class.

        Args:
            api_key (str): OpenAI API key
            model_id (str): Model ID to use for completions
            temperature (float): Sampling temperature
            max_tokens (int): Maximum tokens for completion
            max_loops (int): Maximum number of steps for the BFS algorithm
            breadth_limit (int): Maximum number of states to consider at each level
            number_of_agents (int): Number of thoughts to generate at each step
            log_path (str): Path to save logs
            id (str): Unique identifier for the BFS instance
            LLM_budget (int): Total budget for LLM calls
        """
        # Get API key from environment if not provided
        if api_key is None:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OpenAI API key not provided and not found in environment.")
        
        self.id = id or uuid.uuid4().hex
        self.agent = ToTAgent(
            api_key=api_key,
            model_id=model_id,
            temperature=temperature,
            max_tokens=max_tokens,
            id=self.id
        )
        self.max_loops = max_loops
        self.breadth_limit = breadth_limit
        self.number_of_agents = number_of_agents
        self.log_path = log_path or f"tot_bfs_{self.id}.log"
        self.all_thoughts = []  # Store all thoughts generated during BFS
        self.LLM_budget = LLM_budget
        self.LLM_budget_left = LLM_budget
        
        # Initialize logging
        if self.log_path:
            log_dir = os.path.dirname(os.path.abspath(self.log_path))
            os.makedirs(log_dir, exist_ok=True)
            logger.add(self.log_path, rotation="500 MB")
        
    def _log(self, message: str):
        """Log a message to the log file."""
        logger.info(message)
        
    def bfs(self, state: str, domain: str = "", step_limit: int = -1, results_path: str = None) -> Dict[str, Any]:
        """
        Perform Breadth-First Search (BFS) with a breadth limit based on evaluation scores.

        Args:
            state (str): The initial state or task to explore.
            domain (str): Optional domain context
            step_limit (int): Maximum number of steps to take (-1 for max_loops)
            results_path (str): Path to save step-by-step results

        Returns:
            Dict[str, Any]: The final thought after BFS completes
        """
        # Define maximum steps
        max_steps = self.max_loops if step_limit == -1 else step_limit
        
        # Initialize the set of states
        S = [state]
        
        # Ensure results path directory exists
        if results_path:
            results_path = Path(results_path)
            results_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Clear the file if it exists
            if results_path.exists():
                results_path.unlink()
        
        # Track LLM calls made
        calls_made = 0
        
        # Get initial response (step 0) - direct answer without ToT
        initial_thought = self.agent.run(state, domain, parent_agent=self)
        calls_made += 2
        self._log(f"Initial response (Step 0): Evaluation score = {initial_thought['evaluation']}")
        self._log(f"LLM budget remaining: {self.LLM_budget_left}/{self.LLM_budget}")
        
        # Derive distribution from initial thought
        # CHANGED: Pass parent_agent to count towards budget
        initial_distribution = self.agent.derive_distribution(
            initial_thought["thought"], domain, state, parent_agent=self
        )
        calls_made += 2
        self._log(f"Initial distribution: {initial_distribution}")
        
        # Write ONLY the distribution to the results file - NOT THE THOUGHT
        if results_path:
            with open(results_path, 'w') as f:
                # Write only the distribution list
                f.write(json.dumps(initial_distribution) + '\n')
                self._log(f"Wrote distribution to {results_path}: {json.dumps(initial_distribution)}")
                
        # Store this initial thought (for internal use only)
        self.all_thoughts.append(initial_thought)
        
        # Keep track of the last distribution
        last_distribution = initial_distribution
        
        for t in range(1, max_steps + 1):
            self._log(f"Step {t}/{max_steps}: Expanding states.")
            
            # Check if we've exhausted our budget
            if self.LLM_budget_left <= 0:
                self._log(f"LLM budget exhausted after {calls_made} calls. Stopping.")
                break

            # Generate new thoughts based on current states
            new_states = []
            for s in S:
                if self.LLM_budget_left <= 0:
                    break
                    
                # Generate thoughts using the agent
                for _ in range(min(self.number_of_agents, self.LLM_budget_left)):
                    thought = self.agent.run(s, domain, parent_agent=self)
                    new_states.append((s, thought))
                    calls_made += 2
            
            if not new_states:  # If no new states were generated, stop the BFS
                self._log(f"No valid thoughts generated at step {t} or budget exhausted. Stopping BFS.")
                break

            # Evaluate the new states
            V = [thought["evaluation"] for _, thought in new_states]

            # Log and store all thoughts
            for i, (_, thought) in enumerate(new_states):
                self.all_thoughts.append(thought)
                self._log(f"Thought {i+1}: {thought['thought']} ; (Evaluation: {thought['evaluation']})")

            # Select the best states based on their evaluations, limited by breadth_limit
            state_evaluation_pairs = list(zip(new_states, V))
            state_evaluation_pairs.sort(key=lambda x: x[1], reverse=True)
            best_states = [
                pair[0][1]["thought"]
                for pair in state_evaluation_pairs[: self.breadth_limit]
            ]
            S = best_states
            
            # Find the best thought at this step
            best_thought = None
            if V:
                best_idx = V.index(max(V))
                best_thought = new_states[best_idx][1]
            
            # Derive distribution from best thought at this step
            if best_thought:
                # CHANGED: Check budget before making distribution call and pass parent_agent
                if self.LLM_budget_left > 0:
                    step_distribution = self.agent.derive_distribution(
                        best_thought["thought"], domain, state, parent_agent=self
                    )
                    self._log(f"Step {t} distribution: {step_distribution}")
                    calls_made += 2
                else:
                    # CHANGED: If no budget left, use last distribution
                    step_distribution = last_distribution
                    self._log(f"Step {t} distribution (budget exhausted, using last): {step_distribution}")
                
                # Get current call count (budget used)
                calls_used = self.LLM_budget - self.LLM_budget_left
                
                # Now handle writing to file based on Agent_v2_0 approach
                if results_path:
                    try:
                        # Load existing distributions
                        with open(results_path, 'r') as f:
                            existing = [json.loads(line) for line in f.readlines()]
                    except (FileNotFoundError, json.JSONDecodeError):
                        existing = []
                    
                    # Calculate how many lines we've already written
                    written = len(existing)
                    
                    # Fill gaps with last distribution
                    with open(results_path, 'a') as f:
                        # Fill in distributions for any calls made since last update
                        for i in range(written, calls_used):
                            f.write(json.dumps(last_distribution) + '\n')
                            self._log(f"Filled gap with last distribution at position {i+1}")
                        
                        # Write the new distribution
                        if written < calls_used + 1:  # Only write if we haven't already
                            f.write(json.dumps(step_distribution) + '\n')
                            self._log(f"Wrote new distribution at position {calls_used+1}")
                
                # Update the last distribution
                last_distribution = step_distribution
            
            self._log(f"Completed step {t}. Best evaluation: {max(V) if V else 'N/A'}")
            self._log(f"LLM budget remaining: {self.LLM_budget_left}/{self.LLM_budget}")

        # Fill remaining entries in the results file with the last distribution
        if results_path:
            try:
                # Count existing lines in the file
                with open(results_path, 'r') as f:
                    existing_count = sum(1 for _ in f)
            except FileNotFoundError:
                existing_count = 0
                
            # Fill remaining entries up to budget
            if existing_count < self.LLM_budget:
                with open(results_path, 'a') as f:
                    for _ in range(existing_count, self.LLM_budget):
                        f.write(json.dumps(last_distribution) + '\n')
                        self._log(f"Filled final entry with last distribution")
                    
        # Return the best final thought
        final_answer = None
        if S:
            # Only make this call if we have budget left
            if self.LLM_budget_left > 0:
                # Run the agent on the best state
                final_answer = self.agent.run(S[0], domain, parent_agent=self)
            else:
                # Just use the last best thought
                final_answer = best_thought if 'best_thought' in locals() else initial_thought
                
        return final_answer
    
    def entrypoint(self, question: str, domain: str, step_limit: int, output_path: str, LLM_budget: int = None) -> Dict[str, Any]:
        """
        Main entrypoint function to run Tree of Thoughts BFS reasoning.
        
        This function:
        1. Sets up the agent with the template containing {question} and {domain} placeholders
        2. Runs the BFS algorithm for the specified number of steps or until budget is exhausted
        3. Writes step-by-step results to the specified output file (one line per step)
        
        Args:
            question (str): The question to answer
            domain (str): The domain context
            step_limit (int): Maximum number of steps to take
            output_path (str): Path to save step-by-step results
            LLM_budget (int): Budget for LLM calls (if None, uses the agent's default)
            
        Returns:
            Dict[str, Any]: Final result with complete reasoning path
        """
        # Update budget if provided
        if LLM_budget is not None:
            self.LLM_budget = LLM_budget
            self.LLM_budget_left = LLM_budget
        
        # Log key information
        self._log(f"Starting Tree of Thoughts BFS with {step_limit} step limit and {self.LLM_budget} LLM budget")
        self._log(f"Question: {question}")
        self._log(f"Domain: {domain}")
        self._log(f"Output path: {output_path}")
        
        # Run the BFS algorithm
        final_thought = self.bfs(question, domain, step_limit, output_path)
        
        # Return the final result
        return {
            "question": question,
            "domain": domain,
            "final_thought": final_thought,
            "all_thoughts": self.all_thoughts,
            "llm_calls_made": self.LLM_budget - self.LLM_budget_left
        }


def main():
    """Command line interface for running the ToT BFS agent."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Tree of Thoughts BFS Agent")
    parser.add_argument("--question", type=str, required=True, help="Question to solve")
    parser.add_argument("--domain", type=str, required=True, help="Domain context")
    parser.add_argument("--steps", type=int, required=True, help="Maximum steps")
    parser.add_argument("--output", type=str, required=True, help="Path for step-by-step results")
    parser.add_argument("--api-key", type=str, required=True, help="OpenAI API key")
    parser.add_argument("--model", type=str, default="gpt-4o-2024-08-06", help="Model ID")
    parser.add_argument("--breadth", type=int, default=3, help="Breadth limit")
    parser.add_argument("--agents", type=int, default=3, help="Number of agents per step")
    parser.add_argument("--temp", type=float, default=0.0, help="Temperature")
    parser.add_argument("--budget", type=int, default=30, help="LLM call budget")
    
    args = parser.parse_args()
    
    # Initialize the agent
    agent = ToTBFSAgent(
        api_key=args.api_key,
        model_id=args.model,
        breadth_limit=args.breadth,
        number_of_agents=args.agents,
        temperature=args.temp,
        LLM_budget=args.budget
    )
    
    # Run the ToT inference
    result = agent.entrypoint(
        question=args.question,
        domain=args.domain,
        step_limit=args.steps,
        output_path=args.output,
        LLM_budget=args.budget
    )
    
    # Print final evaluation score
    if result["final_thought"]:
        print(f"Final evaluation score: {result['final_thought']['evaluation']}")
        print(f"LLM calls made: {result['llm_calls_made']}")
    else:
        print("No final thought generated.")


if __name__ == "__main__":
    main()