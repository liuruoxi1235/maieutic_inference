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


def run_experiment(agent_cls, mode, config_path: Path, input_str: str, budget: int, llm_log: Path):
    llm_log.parent.mkdir(parents=True, exist_ok=True)
    # remove stale if exists
    if llm_log.exists():
        llm_log.unlink()
    # instantiate and run agent
    agent = agent_cls(str(config_path), input_str, str(llm_log.parent / 'agent.log'))
    if mode == 'full':
        agent.s_agent_entry(
            max_step=1000,
            allow_human=False,
            LLM_budget=budget,
            LLM_budget_logpth=str(llm_log)
        )
    else:
        agent.s_agent_entry_sampling(
            num_samples=budget,
            allow_human=False,
            LLM_budget=budget,
            LLM_budget_logpth=str(llm_log)
        )
    # read distributions
    distributions = []
    with open(llm_log, 'r') as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            try:
                parsed = json.loads(line)
                if isinstance(parsed, list):
                    distributions.append(parsed)
            except json.JSONDecodeError:
                continue
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


def read_tot_distributions_from(log_path: Path):
    """Read distributions from ToT log file."""
    return read_distributions_from(log_path)  # Our ToT agent outputs in the same format


def main():
    parser = argparse.ArgumentParser(
        description="Compare Agent v1.1 (full) vs v2.0 (sampling) vs ToT with run-by-run plots"
    )
    parser.add_argument("config",       type=Path, help="Agent config JSON file")
    parser.add_argument("settings",     type=Path, help="Multi-line settings file (e.g., 2.txt)")
    parser.add_argument("--budget",     type=int,    default=30,  help="LLM call budget for all agents")
    parser.add_argument("--log-dir",    type=Path,   default=Path("./logs"), help="Base directory for logs")
    parser.add_argument("--output-plot",type=Path,   default=Path("comparison.png"), help="Base plot filename (.png)")
    args = parser.parse_args()

    args.log_dir.mkdir(parents=True, exist_ok=True)
    settings = [l for l in args.settings.read_text().splitlines() if l.strip()]

    cumulative_kl1, cumulative_kl2, cumulative_kl_tot = [], [], []

    for idx, line in enumerate(settings):
        ts = datetime.now().strftime("%Y%m%dT%H%M%S")
        
        # Parse the input line to get question, domains, and ground truth
        try:
            question, domains, ground_truth = parse_input_line(line)
        except ValueError:
            # If parsing fails, fall back to the original parsing
            ground_truth = parse_ground_truth(line)
            question, domains = "", []  # Placeholder values
        
        # Agent 1: locate any existing llm logs
        existing1 = sorted(args.log_dir.glob(f"agent1_{idx}_*.log"), key=lambda p: p.stat().st_mtime, reverse=True)
        if existing1:
            print(f"agent 1 line {idx} is already processed.")
            llm_log1 = existing1[0]
            d1 = read_distributions_from(llm_log1)
        else:
            llm_log1 = args.log_dir / f"agent1_{idx}_{ts}.log"
            d1 = run_experiment(AgentV11, 'full', args.config, line, args.budget, llm_log1)

        # Agent 2: locate any existing llm logs
        existing2 = sorted(args.log_dir.glob(f"agent2_{idx}_*.log"), key=lambda p: p.stat().st_mtime, reverse=True)
        if existing2:
            print(f"agent 2 line {idx} is already processed.")
            llm_log2 = existing2[0]
            d2 = read_distributions_from(llm_log2)
        else:
            llm_log2 = args.log_dir / f"agent2_{idx}_{ts}.log"
            d2 = run_experiment(AgentV20, 'sampling', args.config, line, args.budget, llm_log2)
        
        # Tree of Thoughts: locate any existing logs
        existing_tot = sorted(args.log_dir.glob(f"agent_tot_{idx}_*.log"), key=lambda p: p.stat().st_mtime, reverse=True)
        if existing_tot and question:  # Only use existing if we have parsed question
            print(f"agent tot line {idx} is already processed.")
            llm_log_tot = existing_tot[0]
            d_tot = read_tot_distributions_from(llm_log_tot)
        elif question and domains:  # Only run ToT if we have question and domains
            print(f"running agent tot line {idx}.")
            llm_log_tot = args.log_dir / f"agent_tot_{idx}_{ts}.log"
            d_tot = run_tot_experiment(args.config, question, domains, args.budget, llm_log_tot)
        else:
            print(f"error with tot line {idx}.")
            d_tot = []

        # Compute KL for all methods
        gt = ground_truth
        kl1 = [kl_divergence(gt, dist) for dist in d1] or [0.0]
        kl2 = [kl_divergence(gt, dist) for dist in d2] or [0.0]
        kl_tot = [kl_divergence(gt, dist) for dist in d_tot] or [0.0]

        # Save KL to JSON for caching
        kl1_json = args.log_dir / f"kl_agent1_{idx}.json"
        kl2_json = args.log_dir / f"kl_agent2_{idx}.json"
        kl_tot_json = args.log_dir / f"kl_agent_tot_{idx}.json"
        kl1_json.write_text(json.dumps(kl1))
        kl2_json.write_text(json.dumps(kl2))
        kl_tot_json.write_text(json.dumps(kl_tot))

        cumulative_kl1.append(kl1)
        cumulative_kl2.append(kl2)
        cumulative_kl_tot.append(kl_tot)

        # Compute stats on all runs so far
        max_calls = max(
            max(len(r) for r in cumulative_kl1), 
            max(len(r) for r in cumulative_kl2),
            max(len(r) for r in cumulative_kl_tot) if cumulative_kl_tot else 0
        )
        padded1 = np.array([r + [r[-1]] * (max_calls - len(r)) for r in cumulative_kl1])
        padded2 = np.array([r + [r[-1]] * (max_calls - len(r)) for r in cumulative_kl2])
        
        # Only pad ToT results if we have any
        if cumulative_kl_tot:
            padded_tot = np.array([r + [r[-1]] * (max_calls - len(r)) for r in cumulative_kl_tot])
        
        xs = np.arange(1, max_calls + 1)

        def stats(A): return A.mean(axis=0), np.percentile(A, 25, axis=0), np.percentile(A, 75, axis=0)
        m1, lo1, hi1 = stats(padded1)
        m2, lo2, hi2 = stats(padded2)
        
        # Only compute ToT stats if we have data
        if cumulative_kl_tot:
            m_tot, lo_tot, hi_tot = stats(padded_tot)

        # Per-run plot
        plt.figure()
        plt.plot(xs, m1, label="Agent v1.1 (full)")
        plt.fill_between(xs, lo1, hi1, alpha=0.2)
        plt.plot(xs, m2, label="Agent v2.0 (sampling)")
        plt.fill_between(xs, lo2, hi2, alpha=0.2)
        
        # Add ToT line if we have data
        if cumulative_kl_tot:
            plt.plot(xs, m_tot, 'r-', label="Tree of Thoughts")
            plt.fill_between(xs, lo_tot, hi_tot, color='r', alpha=0.2)
        
        plt.xlabel("LLM Call Count")
        plt.ylabel("KL Divergence")
        plt.legend()
        plt.grid(True)

        stem, ext = args.output_plot.stem, args.output_plot.suffix
        out_plot = args.output_plot.with_name(f"{stem}_{idx}{ext}")
        plt.savefig(out_plot)
        plt.close()

    # Final cumulative plot
    final_plot = args.output_plot.with_name(f"{args.output_plot.stem}_final{args.output_plot.suffix}")
    plt.figure()
    plt.plot(xs, m1, label="Agent v1.1 (full)")
    plt.fill_between(xs, lo1, hi1, alpha=0.2)
    plt.plot(xs, m2, label="Agent v2.0 (sampling)")
    plt.fill_between(xs, lo2, hi2, alpha=0.2)
    
    # Add ToT line if we have data
    if cumulative_kl_tot:
        plt.plot(xs, m_tot, 'r-', label="Tree of Thoughts")
        plt.fill_between(xs, lo_tot, hi_tot, color='r', alpha=0.2)  # Add confidence band for ToT
    
    plt.xlabel("LLM Call Count")
    plt.ylabel("KL Divergence")
    plt.legend()
    plt.grid(True)
    plt.savefig(final_plot)
    plt.close()

if __name__ == "__main__":
    main()