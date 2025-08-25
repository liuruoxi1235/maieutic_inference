import argparse
import ast
import json
import sys
import re
import os
from pathlib import Path
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Optional

# Ensure script directory is in sys.path for imports
sys.path.insert(0, str(Path(__file__).parent))
import Agent_v1_1
import Agent_v2_0
import Agent_ToT  # Our ToT implementation

# Load environment variables for API keys
from dotenv import load_dotenv
load_dotenv()

AgentV11 = Agent_v1_1.AgentS1
AgentV20 = Agent_v2_0.AgentS1
ToTBFSAgent = Agent_ToT.ToTBFSAgent  # Our ToT agent

def parse_ground_truth(line: str):
    start = line.rfind('[')
    end = line.rfind(']')
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"Cannot parse ground truth from line: {line}")
    return ast.literal_eval(line[start:end+1])


def parse_input_line(line: str):
    """Parse input line to extract question, domains, and ground truth distribution."""
    parts = line.split(';')
    if len(parts) < 3:
        raise ValueError(f"Invalid input format: {line}")
    
    question = parts[0].strip()
    
    # Parse domains - handle JSON format instead of Python literal
    domains_str = parts[1].strip()
    try:
        # First try with json.loads which handles strings with spaces better
        domains = json.loads(domains_str)
    except json.JSONDecodeError:
        # Fallback to the original method
        domains_start = domains_str.find('[')
        domains_end = domains_str.rfind(']')
        if domains_start == -1 or domains_end == -1 or domains_end <= domains_start:
            raise ValueError(f"Cannot parse domains from: {domains_str}")
        
        # Clean up the domain string to handle spaces
        domain_content = domains_str[domains_start+1:domains_end]
        domain_items = []
        for item in domain_content.split(','):
            # Remove extra whitespace and quotes
            cleaned = item.strip()
            if cleaned.startswith('"') and cleaned.endswith('"'):
                cleaned = cleaned[1:-1]
            elif cleaned.startswith("'") and cleaned.endswith("'"):
                cleaned = cleaned[1:-1]
            domain_items.append(cleaned)
        domains = domain_items
    
    # Parse ground truth distribution
    gt_str = parts[2].strip()
    gt_start = gt_str.find('[')
    gt_end = gt_str.rfind(']')
    if gt_start == -1 or gt_end == -1 or gt_end <= gt_start:
        raise ValueError(f"Cannot parse ground truth from: {gt_str}")
    
    try:
        ground_truth = json.loads(gt_str[gt_start:gt_end+1])
    except json.JSONDecodeError:
        # Fallback to ast.literal_eval
        ground_truth = ast.literal_eval(gt_str[gt_start:gt_end+1])
    
    return question, domains, ground_truth


def kl_divergence(p, q, eps=1e-12):
    p = np.array(p, dtype=float)
    q = np.array(q, dtype=float)
    q = np.clip(q, eps, None)
    return float(np.sum(p * np.log(p / q)))


def should_filter_kl_file(kl_values: List[float], threshold: float = 0.02, threshold_2: float = 1.00, max_steps: int = 3) -> bool:
    """
    Check if a KL divergence file should be filtered out.
    
    Args:
        kl_values: List of KL divergence values
        threshold: KL threshold below which steps are considered "too good"
        max_steps: Maximum number of steps allowed below threshold
        
    Returns:
        True if the file should be filtered out (excluded from plots)
    """
    if not kl_values:
        return False
    
    # Count how many steps have KL < threshold
    steps_below_threshold = sum(1 for kl in kl_values if kl < threshold)
    steps_above_threshold = sum(1 for kl in kl_values if kl > threshold_2)
    
    # Filter out if more than max_steps have KL < threshold
    return (steps_below_threshold > max_steps) or (steps_above_threshold > max_steps)


def run_agent_and_compute_kl(agent_name: str, config_paths: Dict, line: str, question: str, domains: List[str], budget: int, llm_log: Path, ground_truth: List[float], log_dir: Path, idx: int) -> tuple:
    """
    Run an agent and compute KL divergence values.
    
    Returns:
        tuple: (distributions, kl_values)
    """
    # Run the appropriate agent
    if agent_name == "tot":
        if question and domains:
            distributions = run_tot_experiment(None, question, domains, budget, llm_log)
        else:
            print(f"Skipping ToT for line {idx} - missing question/domains")
            distributions = []
    elif agent_name == "agent_1":
        distributions = run_agent_1_experiment(config_paths["agent_1"], line, budget, llm_log)
    elif agent_name == "agent_2_s":
        distributions = run_agent_2_s_experiment(config_paths["agent_2_s"], line, budget, llm_log)
    elif agent_name == "agent_2_nl":
        distributions = run_agent_2_nl_experiment(config_paths["agent_2_nl"], line, budget, llm_log)
    else:
        distributions = []
    
    # Compute and save KL divergence
    kl_values = [kl_divergence(ground_truth, dist) for dist in distributions] or [0.0]
    kl_json = log_dir / f"kl_{agent_name}_{idx}.json"
    kl_json.write_text(json.dumps(kl_values))
    
    return distributions, kl_values


def run_agent_1_experiment(config_path: Path, input_str: str, budget: int, llm_log: Path):
    """Run Agent v1.1 (full exploration) experiment."""
    llm_log.parent.mkdir(parents=True, exist_ok=True)
    if llm_log.exists():
        llm_log.unlink()
    
    agent = AgentV11(str(config_path), input_str, str(llm_log.parent / 'agent1.log'))
    agent.s_agent_entry(
        max_step=1000,
        allow_human=False,
        LLM_budget=budget,
        LLM_budget_logpth=str(llm_log)
    )
    
    return read_distributions_from(llm_log)


def run_agent_2_s_experiment(config_path: Path, input_str: str, budget: int, llm_log: Path):
    """Run Agent v2.0 structured sampling experiment."""
    llm_log.parent.mkdir(parents=True, exist_ok=True)
    if llm_log.exists():
        llm_log.unlink()
    
    agent = AgentV20(str(config_path), input_str, str(llm_log.parent / 'agent2_s.log'))
    agent.s_agent_entry_sampling(
        num_samples=budget,
        allow_human=False,
        LLM_budget=budget,
        LLM_budget_logpth=str(llm_log)
    )
    
    return read_distributions_from(llm_log)


def run_agent_2_nl_experiment(config_path: Path, input_str: str, budget: int, llm_log: Path):
    """Run Agent v2.0 natural language sampling experiment."""
    llm_log.parent.mkdir(parents=True, exist_ok=True)
    if llm_log.exists():
        llm_log.unlink()
    
    agent = AgentV20(str(config_path), input_str, str(llm_log.parent / 'agent2_nl.log'))
    agent.nl_agent_entry_sampling(
        num_samples=budget,
        allow_human=False,
        LLM_budget=budget,
        LLM_budget_logpth=str(llm_log)
    )
    
    return read_distributions_from(llm_log)


def run_tot_experiment(config_path: Path, question: str, domains: List[str], budget: int, log_path: Path):
    """Run Tree of Thoughts reasoning experiment using our ToTBFSAgent."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if log_path.exists():
        log_path.unlink()
    
    # Create a debug log file
    debug_log_path = log_path.with_suffix('.debug.log')
    with open(debug_log_path, 'w') as f:
        f.write(f"Running ToT experiment\n")
        f.write(f"Question: {question}\n")
        f.write(f"Domains: {domains}\n")
        f.write(f"Budget: {budget}\n")
        f.write(f"Log path: {log_path}\n")
    
    try:
        # Format domain as a string
        domain_str = json.dumps(domains)
        
        # Initialize our ToTBFSAgent
        tot_agent = ToTBFSAgent(
            model_id="gpt-4o-2024-08-06",   # Using the default model
            max_loops=budget,               # Configure to match budget
            breadth_limit=3,                # Standard breadth limit
            number_of_agents=3,             # Standard number of agents
            log_path=str(debug_log_path),   # Use the same debug log
            LLM_budget=budget               # Set the LLM call budget
        )
        
        with open(debug_log_path, 'a') as f:
            f.write("Initialized ToTBFSAgent\n")
        
        # Run the ToT BFS algorithm via the entrypoint
        result = tot_agent.entrypoint(
            question=question,
            domain=domain_str,
            step_limit=budget,
            output_path=str(log_path),
            LLM_budget=budget
        )
        
        with open(debug_log_path, 'a') as f:
            f.write(f"ToT execution completed\n")
            f.write(f"LLM calls made: {result.get('llm_calls_made', 'unknown')}\n")
            f.write(f"Final evaluation: {result['final_thought']['evaluation'] if result['final_thought'] else 'None'}\n")
        
    except Exception as e:
        # Log any errors that might occur
        with open(debug_log_path, 'a') as f:
            f.write(f"Error during ToT execution: {str(e)}\n")
            import traceback
            f.write(traceback.format_exc())
        
        # Create empty file in case of error
        with open(log_path, 'w') as f:
            f.write(json.dumps({"error": str(e)}) + "\n")
        return []
    
    # Read the distributions back from the file
    distributions = read_distributions_from(log_path)
    
    # If we don't have enough distributions to match the budget, pad with the last one
    if distributions and len(distributions) < budget:
        last_dist = distributions[-1]
        distributions.extend([last_dist] * (budget - len(distributions)))
    
    return distributions


def read_distributions_from(log_path: Path):
    dists = []
    with open(log_path, 'r') as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            try:
                parsed = json.loads(line)
                if isinstance(parsed, list):
                    dists.append(parsed)
            except json.JSONDecodeError:
                continue
    return dists


def load_config(config_path: Path) -> Dict[str, Any]:
    """Load experiment configuration from JSON file."""
    with open(config_path, 'r') as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(
        description="Compare multiple agents with configurable selection"
    )
    parser.add_argument("config", type=Path, help="Experiment config JSON file")
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Extract configuration parameters
    config_paths = {
        "agent_1": Path(config["agent_1_config"]) if "agent_1_config" in config else None,
        "agent_2_s": Path(config["agent_2_s_config"]) if "agent_2_s_config" in config else None,
        "agent_2_nl": Path(config["agent_2_nl_config"]) if "agent_2_nl_config" in config else None,
    }
    input_file = Path(config["input_file"])
    budget = config["budget"]
    log_dir = Path(config["log_dir"])
    output_plot_base = Path(config["output_plot"])
    enabled_agents = config["enabled_agents"]  # List of agent names to run
    
    # Validate enabled agents and their configs
    valid_agents = {"agent_2_nl", "agent_2_s", "tot", "agent_1"}
    for agent in enabled_agents:
        if agent not in valid_agents:
            raise ValueError(f"Invalid agent '{agent}'. Valid options: {valid_agents}")
        # Only validate config paths for non-ToT agents
        if agent != "tot" and config_paths.get(agent) is None:
            raise ValueError(f"Config file not specified for enabled agent '{agent}'")
    
    log_dir.mkdir(parents=True, exist_ok=True)
    input_lines = [l for l in input_file.read_text().splitlines() if l.strip()]

    # Store cumulative KL divergences for each enabled agent
    cumulative_kls = {agent: [] for agent in enabled_agents}
    agent_labels = {
        "tot": "Tree of Thoughts",
        "agent_1": "Agent v1.1 (full)",
        "agent_2_s": "Agent v2.0 (structured)",
        "agent_2_nl": "Agent v2.0 (natural language)"
    }
    agent_colors = {
        "tot": "red",
        "agent_1": "blue", 
        "agent_2_s": "green",
        "agent_2_nl": "orange"
    }

    for idx, line in enumerate(input_lines):
        print("######################################################")
        print(f"###############working on line {idx} ################")
        print("######################################################")
        ts = datetime.now().strftime("%Y%m%dT%H%M%S")
        
        # Parse the input line to get question, domains, and ground truth
        try:
            question, domains, ground_truth = parse_input_line(line)
        except ValueError:
            # If parsing fails, fall back to the original parsing
            ground_truth = parse_ground_truth(line)
            question, domains = "", []  # Placeholder values
        
        # Run each enabled agent
        agent_results = {}
        
        for agent_name in enabled_agents:
            print("######################################################")
            print(f"###############working on agent {agent_name} ########")
            print("######################################################")
            # Look for existing prediction files (exclude debug files)
            existing_files = sorted(
                [f for f in log_dir.glob(f"{agent_name}_{idx}_*.log") if '.debug.' not in f.name], 
                key=lambda p: p.stat().st_mtime, 
                reverse=True
            )
            
            kl_values = None  # Initialize KL values
            distributions = []  # Initialize distributions
            
            # Step 1: Check if KL file already exists
            kl_json = log_dir / f"kl_{agent_name}_{idx}.json"
            if kl_json.exists():
                print(f"{agent_name} line {idx} KL file already exists.")
                try:
                    with open(kl_json, 'r') as f:
                        kl_values = json.loads(f.read())
                    
                    # Also need to load distributions for agent_results
                    if existing_files:
                        llm_log = existing_files[0]
                        distributions = read_distributions_from(llm_log)
                    else:
                        # KL exists but no prediction file - this is unusual but handle it
                        print(f"Warning: KL file exists but no prediction file for {agent_name} line {idx}")
                        distributions = []
                        
                except Exception as e:
                    print(f"Error loading KL file for {agent_name} line {idx}: {str(e)}")
                    kl_values = None
            
            # Step 2: If no KL file, check for existing prediction files
            if kl_values is None and existing_files:
                print(f"{agent_name} line {idx} prediction file exists, generating KL.")
                llm_log = existing_files[0]
                distributions = read_distributions_from(llm_log)
                if distributions:
                    # Generate KL from existing predictions
                    kl_values = [kl_divergence(ground_truth, dist) for dist in distributions]
                    # Save the generated KL file for future use
                    kl_json.write_text(json.dumps(kl_values))
                    print(f"Generated and saved KL file for {agent_name} line {idx}")
                else:
                    print(f"Could not read distributions from {llm_log}, need to re-run agent.")
            
            # Step 3: If still no KL values, run the agent
            if kl_values is None:
                print(f"Running {agent_name} line {idx} from scratch.")
                llm_log = log_dir / f"{agent_name}_{idx}_{ts}.log"
                distributions, kl_values = run_agent_and_compute_kl(agent_name, config_paths, line, question, domains, budget, llm_log, ground_truth, log_dir, idx)
            
            agent_results[agent_name] = distributions
            
            # Filter out KL files based on the criteria (only if we have kl_values)
            if kl_values is not None:
                if should_filter_kl_file(kl_values):
                    print(f"Filtering out {agent_name} line {idx} - more than 3 steps with KL < 0.02")
                    # Don't add to cumulative_kls, effectively excluding from plots
                else:
                    cumulative_kls[agent_name].append(kl_values)

        # Generate per-run plot
        if any(cumulative_kls.values()):
            max_calls = max(max(len(r) for r in agent_kls) for agent_kls in cumulative_kls.values() if agent_kls)
            xs = np.arange(1, max_calls + 1)
            
            plt.figure(figsize=(10, 6))
            
            for agent_name in enabled_agents:
                if cumulative_kls[agent_name]:
                    # Pad results to match max_calls
                    padded = np.array([r + [r[-1]] * (max_calls - len(r)) for r in cumulative_kls[agent_name]])
                    mean_kl = padded.mean(axis=0)
                    q25 = np.percentile(padded, 25, axis=0)
                    q75 = np.percentile(padded, 75, axis=0)
                    
                    color = agent_colors[agent_name]
                    label = agent_labels[agent_name]
                    
                    plt.plot(xs, mean_kl, color=color, label=label)
                    
                    # Only show shaded region for agent_2_s and agent_2_nl
                    if agent_name in ["agent_2_s", "agent_2_nl"]:
                        plt.fill_between(xs, q25, q75, color=color, alpha=0.2)
            
            plt.xlabel("LLM Call Count")
            plt.ylabel("KL Divergence")
            plt.legend()
            plt.grid(True)
            plt.title(f"Agent Comparison - {idx+1} runs")
            
            # Save per-run plot
            stem, ext = output_plot_base.stem, output_plot_base.suffix
            out_plot = output_plot_base.with_name(f"{stem}_{idx}{ext}")
            plt.savefig(out_plot, dpi=150, bbox_inches='tight')
            plt.close()

    # Generate final cumulative plot
    if any(cumulative_kls.values()):
        max_calls = max(max(len(r) for r in agent_kls) for agent_kls in cumulative_kls.values() if agent_kls)
        xs = np.arange(1, max_calls + 1)
        
        plt.figure(figsize=(12, 8))
        
        for agent_name in enabled_agents:
            if cumulative_kls[agent_name]:
                # Pad results to match max_calls
                padded = np.array([r + [r[-1]] * (max_calls - len(r)) for r in cumulative_kls[agent_name]])
                mean_kl = padded.mean(axis=0)
                q25 = np.percentile(padded, 25, axis=0)
                q75 = np.percentile(padded, 75, axis=0)
                
                color = agent_colors[agent_name]
                label = agent_labels[agent_name]
                
                plt.plot(xs, mean_kl, color=color, label=label, linewidth=2)
                
                # Only show shaded region for agent_2_s and agent_2_nl
                if agent_name in ["agent_2_s", "agent_2_nl"]:
                    plt.fill_between(xs, q25, q75, color=color, alpha=0.2)
        
        plt.xlabel("LLM Call Count")
        plt.ylabel("KL Divergence")
        plt.legend()
        plt.grid(True)
        plt.title(f"Final Agent Comparison - {len(input_lines)} runs")
        
        # Save final plot
        final_plot = output_plot_base.with_name(f"{output_plot_base.stem}_final{output_plot_base.suffix}")
        plt.savefig(final_plot, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Final comparison plot saved to: {final_plot}")


if __name__ == "__main__":
    main()