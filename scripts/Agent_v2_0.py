from pathlib import Path
import argparse
import json
import dotenv
import json
import numpy as np
import openai
import tempfile
import re
import ast
import math
from collections import deque
from abc import ABC, abstractmethod
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
from statsmodels.graphics.mosaicplot import mosaic
import os
import random
from copy import deepcopy
import copy
import argparse
import json
from pathlib import Path

dotenv.load_dotenv('.env')

class AgentS1():
    def __init__(self, config, input, log_pth='./example_files'):
        with open(config, 'r') as config_file:
            schema = json.load(config_file)

        self.LLM_budget = 100000
        self.LLM_budget_left = 100000

        #storing if we are working with structured data or natural language data
        self.structured = schema["agent"]["structured"] 

        #model we prefer to use
        self.model_ID = schema["agent"]["model_dial"]

        #temperature
        self.temperature = schema["agent"]["temperature"]

        #the schema
        self.schema = schema["agent"]["structured_prompts"]["node_schema"]
        self.gss_df = None

        self.INVALID_RESPONSES = {"don't know", "iap", "not available in this year", "no answer", "skipped on web", "refused"}

        #logging
        log_pth = Path(log_pth)
        log_pth.parent.mkdir(parents=True, exist_ok=True)
        log_pth.touch(exist_ok=True)
        self.log_pth = log_pth

        '''step_hist_pth = Path(schema["agent"]["step_hist"])
        step_hist_pth.parent.mkdir(parents=True, exist_ok=True)
        step_hist_pth.touch(exist_ok=True)
        self.step_hist_pth = step_hist_pth'''

        self.traversed_vars = set()

        #prompts
        self.natural_prompts = schema["agent"]["natural_prompts"]
        self.structured_prompts = schema["agent"]["structured_prompts"]

        #chat history
        if "history" in schema:
            self.history = schema["history"]
        else:
            self.history = dict()

        #starting point
        self.nl_root = schema["root"]
        with open(self.log_pth, "a") as log_file:
                    log_file.write(f"[main 0] [Step 0] : Starting with natural language query: {self.nl_root} \n")

        #initialize a queue for unprocessed nodes (leaves)
        self.queue = deque()

        self.LLM_call_logpath = "./Sampling_tree_agent_answers.txt"

        #initialize openai api
        import os
        from dotenv import load_dotenv
        load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY")
        openai.api_key = api_key
        self.llm = openai

        if self.structured:
            self.s_root = self.init_full_root_from_txt(input)

            temp_root = copy.deepcopy(self.s_root)
            self.craft_target_distribution_S(temp_root)
            self.root_estimate = temp_root.target.prob
            
            self.tree = STree(self.s_root)
            self.queue.append(self.s_root)
        else:
            self.nl_root = self.init_full_root_from_txt_nl(input)

            temp_root = copy.deepcopy(self.nl_root)
            self.craft_target_distribution_NL(temp_root)
            self.root_estimate = temp_root.target.prob

            self.tree = NLTree(self.nl_root)
            self.queue.append(self.nl_root)

        
        with open(self.log_pth, "a") as log_file:
            log_file.write(f"[main 0] [step 0] : Initialization finished. Now going to take steps after decisions. \n")
            if self.structured:
                log_file.write(f"[main 0] [content] : {json.dumps(self.s_root.to_json())} \n")
                print(f"[main 0] [content] : {json.dumps(self.s_root.to_json())} \n")
            else:
                log_file.write(f"[main 0] [content] : {json.dumps(self.nl_root.to_json())} \n")
                print(f"[main 0] [content] : {json.dumps(self.nl_root.to_json())} \n")
            

    

    def init_root_from_txt(self, input):
        chosen_line = input
        # Initialize a target Variable using the chosen line.
        # Instead of providing an empty list for prob, we pass None so that no probability validation is triggered.
        target_var = Variable(name=chosen_line, type="target", value=[], prob=None)
        # Create an S_node with:
        # - name set to "0"
        # - target set to the above Variable,
        # - empty unbound and bound lists,
        # - and a default probability of 1.
        root_node = S_node(name="0", target=target_var, unbound=[], bound=[], prob=1)

        return root_node
    
    def init_full_root_from_txt(self, input_str):
        """
        input_str format:
          "<question_text>; [val1, val2, ...]; [p1; p2; ...]"
        - First part is the target variable name (string).
        - Second part is a Python‐style list of domain values, comma‐separated.
        - Third part is a Python‐style list of probabilities, semicolon‐separated.
        Returns an S_node with that target variable fully specified.
        """
        # split into three parts only
        parts = input_str.split(';', 2)
        if len(parts) != 3:
            raise ValueError(f"Expected 3 parts separated by ';', got: {input_str!r}")

        # 1) name
        name = parts[0].strip()

        # 2) values: strip [ ] then split on commas
        vals_part = parts[1].strip().lstrip('[').rstrip(']')
        values = [v.strip() for v in vals_part.split(',') if v.strip()]


        # build the target Variable and root S_node
        target_var = Variable(name=name, type="target", value=values, prob=None)
        root_node = S_node(name="0", target=target_var, unbound=[], bound=[], prob=1.0)
        return root_node
    
    def init_full_root_from_txt_nl(self, input_str):
        """
        Initialize NL root node from input string with domain parsing.
        input_str format: "<question_text>; [val1, val2, ...]; [p1; p2; ...]"
        - First part is the target question text (string).
        - Second part is a Python-style list of domain values, comma-separated.
        - Third part is ignored (probabilities will be estimated by LLM later).
        Returns an NL_node with target question, domain, and initial facts parsed from LLM response.
        """
        # Parse the three-component input format like the structured version
        parts = input_str.split(';', 2)
        if len(parts) != 3:
            raise ValueError(f"Expected 3 parts separated by ';', got: {input_str!r}")

        # 1) Extract question text
        question_text = parts[0].strip()

        # 2) Extract domain values: strip [ ] then split on commas
        vals_part = parts[1].strip().lstrip('[').rstrip(']')
        domain_values = [v.strip() for v in vals_part.split(',') if v.strip()]

        # 3) Third part (probabilities) is ignored - will be estimated by LLM later
        
        # Use LLM to parse initial facts from the question
        init_prompt = self.natural_prompts["init_prompt"]
        
        response = self.single_LLM_call(
            init_prompt,
            self.llm,
            "init_nl_root",
            replacements={"q": question_text}  # Only use the question text for LLM
        )
        
        # Parse response format: "Target question: [What is the weather today?] Known facts: [Yesterday was rainy. ; I lost my umbrella.]"
        try:
            # Extract target question (should match our parsed question_text)
            target_match = re.search(r'Target question:\s*\[([^\]]+)\]', response)
            if target_match:
                llm_question = target_match.group(1).strip()
                # Use the LLM's reformulated question if provided, otherwise use original
                final_question = llm_question if llm_question else question_text
            else:
                # Fallback to original question text if LLM doesn't provide one
                final_question = question_text
            
            # Extract known facts
            facts_match = re.search(r'Known facts:\s*\[([^\]]+)\]', response)
            initial_facts = []
            if facts_match:
                facts_str = facts_match.group(1)
                # Split on semicolon and clean up
                initial_facts = [fact.strip() for fact in facts_str.split(';') if fact.strip()]
            
            # Create target variable with the parsed domain
            target_var = Variable(
                name=final_question, 
                type="target", 
                value=domain_values,  # Use parsed domain values
                prob=None  # Will be estimated later
            )
            
            # Create root node
            root_node = NL_node(
                name="0",
                target=target_var,
                questions=[],  # Start with empty questions
                facts=initial_facts,
                prob=1.0
            )
            
            return root_node
            
        except Exception as e:
            with open(self.log_pth, "a") as log_file:
                log_file.write(f"[err] Error parsing init response: {e}\n")
            raise

    # Helper function to convert a question Variable with defined domain into a list of fact strings
    def convert_question_to_facts(self, question_var):
        """
        Convert a question Variable into a list of declarative fact strings.
        Args:
            question_var: Variable object with name (question text) and value (domain options)
        Returns:
            List of fact strings corresponding to each domain value
        """
        if not question_var.value:
            raise ValueError(f"Question '{question_var.name}' has no defined domain values")
        
        # Use the convert_json_facts prompt to get LLM to convert the question to facts
        convert_prompt = self.natural_prompts["convert_json_facts"]

        # Format the question properly for the prompt
        # Create format: "Question content [domain1, domain2, ...]"
        domain_str = "[" + "; ".join(question_var.value) + "]"  # Use semicolon for domain separation
        formatted_question = f"{question_var.name} {domain_str}"
        
        response = self.single_LLM_call(
            convert_prompt, 
            self.llm, 
            "convert_facts",
            replacements={"q": formatted_question}  # Use "q" placeholder as shown in prompt
        )
        
        # Parse the response which should be in format: [fact1; fact2; ...] (semicolon-separated)
        try:
            # Extract the list content from the response
            start_idx = response.find('[')
            end_idx = response.rfind(']')
            if start_idx != -1 and end_idx != -1:
                list_content = response[start_idx+1:end_idx].strip()
                
                # Split by semicolon and clean up each fact
                facts_list = []
                if list_content:
                    raw_facts = list_content.split(';')  # Split on semicolon instead of comma
                    for fact in raw_facts:
                        # Remove any surrounding quotes and whitespace
                        clean_fact = fact.strip().strip('"').strip("'").strip()
                        if clean_fact:
                            facts_list.append(clean_fact)
                
                # Validate that we got the expected number of facts
                if len(facts_list) != len(question_var.value):
                    with open(self.log_pth, "a") as log_file:
                        log_file.write(f"[err] Expected {len(question_var.value)} facts but got {len(facts_list)} for question '{question_var.name}'\n")
                    return []
                
                return facts_list
            else:
                raise ValueError("Could not find list format in response")
                
        except Exception as e:
            with open(self.log_pth, "a") as log_file:
                log_file.write(f"[err] Could not parse facts conversion: {e}\n")
            return []

    # Helper function to format questions/variables for NL prompts
    def format_variables_for_prompt(self, variables):
        """
        Format a Variable or list of Variables into the required string format:
        "Question content 1 [domain1, domain2] \n Question content 2 \n ..."
        Args:
            variables: Single Variable or list of Variables
        Returns:
            String with formatted questions, one per line
        """
        # Handle single variable case
        if isinstance(variables, Variable):
            variables = [variables]
        
        formatted_lines = []
        for var in variables:
            line = var.name
            if var.value:  # If domain is defined, add it in brackets
                domain_str = "[" + "; ".join(var.value) + "]"
                line += " " + domain_str
            formatted_lines.append(line)
        
        return " \n ".join(formatted_lines)

    # Helper function to format facts for NL prompts
    def format_facts_for_prompt(self, facts):
        """
        Format a list of fact strings into a single newline-separated string.
        Args:
            facts: List of fact strings
        Returns:
            String with facts separated by \n
        """
        if not facts:
            return ""
        return " \n ".join(facts)

    def s_agent_entry(self, max_step = 100, allow_human = False):      
        root = self.tree.root

        #we can either set a finite number of steps to take or let the model decide when to stop
        if max_step != -1:
            step_mode = True 
        
        with open(self.log_pth, "a") as log_file:
            log_file.write(f"[main 0] [step 0] : Starting agentic workflow. Allowing {max_step} steps. \n")
            
        if step_mode:

            step = 0
            additional_msg = None
            stopped_node = []
            while(step < max_step):
                step = step + 1
                # get the front item to work on. Intially this is the root.
                # queue would always store the leaves. When a node bounds a variable, it is dequeued and its children are enqueueed
                node = self.queue[0]

                var = node.ready_to_spawn()
                #here we have an unbound var that has a probability distribution. we should first spawn children
                if var is not None and var.name != self.s_root.target.name:
                    self.queue.popleft()
                    children = self.tree.spawn_children(var)
                    for child in children:
                        self.queue.append(child)

                # let model decide which step to take next:         
                num, var_str = self.decision_S(node, allow_human, additional_msg)
                additional_msg = None

                # 1: propose a new variable
                if num == 1:

                    # this is the node with the variale proposed
                    self.propose_variable_S(node)

                    with open(self.log_pth, "a") as log_file:
                        log_file.write(f"[main {node.name}] [step {step}] : proposed variable (action 1). \n")
                        log_file.write(f"[main {node.name}] [content] : {json.dumps(node.to_json())}\n")
 
                #2: choose the domain for an unbound variable
                elif num == 2:
                    input_var = None
                    for var in node.unbound:
                        if var.name == var_str:
                            input_var = var

                    self.propose_domain_S(node, input_var)

                    with open(self.log_pth, "a") as log_file:
                        log_file.write(f"[main {node.name}] [step {step}] : chose domain (action 2)\n")
                        log_file.write(f"[main {node.name}] [content] : {json.dumps(node.to_json())}\n")

                #3. craft a distribution for an unbound variable 
                elif num == 3:
                    input_var = None
                    for var in node.unbound:
                        if var.name == var_str:
                            input_var = var
                    if input_var == None:
                        with open(self.log_pth, "a") as log_file:
                            log_file.write(f"[err] Did not successfuly found a correct var\n\n")
                            additional_msg = [
                                ("assistant", f"3, {var_str}"),
                                ("user", "Your given variable name is not in the unbound variable list. Return the selection number and the exact name of the variable that appears in the given json. ")
                            ]
                            step = step - 1
                            continue

                    
                    self.craft_distribution_S(node, input_var)

                    # since a variable is fully defined, we need to spawn the childrens and do inference on them
                    children = self.tree.spawn_children(node)
                    self.queue.popleft()
                    for child in children:
                        self.queue.append(child)

                    with open(self.log_pth, "a") as log_file:
                        log_file.write(f"[main {node.name}] [step {step}] : estimated distribution (action 3)\n")
                        log_file.write(f"[main {node.name}] [content] : {json.dumps(node.to_json())}\n")
                    
                
                #4. ask for human definition
                #elif num == 4:
                #    input_var = None
                #    for var in node.unbound:
                #        if var.name == var_str:
                #            input_var = var
                #    node = self.human_question_S(node, input_var)

                #5. stop at this node
                elif num == 4:

                    #if there's only one node left, and we haven't used up the steps, then the LLM is not allowed to stop at this node
                    if len(self.queue) == 1 and step != max_step:
                        step = step - 1
                        additional_msg = [
                            ("assistant", "4"),
                            ("user", "Option 4 is not allowed at this node. Please choose a different action.")
                        ]   
                        continue
                    else:
                        if node.name in stopped_node:
                            additional_msg = [
                                ("assistant", "4"),
                                ("user", "This node has already been stopped. Please choose a different action.")
                            ]
                            continue
                        else:
                            stopped_node.append(node.name)
                            #Finish here
                            self.craft_target_distribution_S(node)
                            node = self.queue.popleft()
                            self.queue.append(node)
                            with open(self.log_pth, "a") as log_file:
                                log_file.write(f"[main {node.name}] [step {step}] : stopped at this node (action 4)\n")
                                log_file.write(f"[main {node.name}] [content] : {json.dumps(node.to_json())}\n")
            
            #after the given number of steps are taken, terminate all currently pending nodes, and trace backward
            while self.queue:
                node = self.queue.popleft()
                self.craft_target_distribution_S(node)
                with open(self.log_pth, "a") as log_file:
                    log_file.write(f"[main {node.name}] [GSS replacement internal steps] : Crafted target distribution at this node to start backtracking. \n")
                    log_file.write(f"[main {node.name}] [content] : {json.dumps(node.to_json())}\n")
            
            self.recursive_backtrack(root)


            print(root.__repr__())


    def s_agent_entry_sampling(self, num_samples=100, allow_human=False, LLM_budget=10000, LLM_budget_logpth = "./1.txt"):
        """
        Build the tree by sampling `num_samples` root-to-leaf paths.
        Each time we clear the previous 'active' flags, then mark the new path.
        """
        self.LLM_budget = LLM_budget
        self.LLM_budget_left = LLM_budget
        self.LLM_call_logpath = LLM_budget_logpth
        print(f"LLM budget: {self.LLM_budget}")

        root = self.tree.root
        with open(self.log_pth, "a") as log_file:
            log_file.write(f"[main sampling] Starting sampling with {num_samples} samples.\n")

        if self.LLM_budget != 10000:
            self.craft_target_distribution_S_uncertain(self.s_root)
            with open(self.LLM_call_logpath, 'w') as wf:
                wf.write(json.dumps(self.s_root.target.prob) + "\n")
                print(f"[!] Writing a line to LLM call pth!")
            print(f"initial target distribution {self.s_root.target.prob} at root node {self.s_root}.")
            self.s_root.target.prob = None

        for i in range(1, num_samples+1):

            if not self.LLM_budget_left > 0:
                break

            with open(self.log_pth, "a") as log_file:
                log_file.write(f"[main sampling {i}] Clearing active flags, starting sample {i}\n")

            # Reset all prior flags
            self.tree.clear_active()

            with open(self.log_pth, "a") as log_file:
                log_file.write(f"[main sampling {i}] Sampling to leaf.\n")
            self.sampling_to_leaf(allow_human=allow_human)

            # After sampling, dump the full tree with active flags
            full_tree_json = repr(self.tree)
            with open(self.log_pth, "a") as log_file:
                log_file.write(f"[main sampling {i}] Full tree JSON:\n{full_tree_json}\n")

            self.recursive_backtrack(root)

            calls_made = self.LLM_budget - self.LLM_budget_left
            print(f"calls made: {calls_made}")
            print(f"self.LLM_budget: {self.LLM_budget}")
            print(f"self.LLM_budget_left: {self.LLM_budget_left}")
            # load what’s already there
            try:
                with open(self.LLM_call_logpath, 'r') as f:
                    existing = [json.loads(line) for line in f]
            except FileNotFoundError:
                existing = []

            written = len(existing)
            last_dist = existing[-1] if existing else None
            curr_dist = root.target.prob

            # fill “gaps” with last_dist up to the *previous* call
            with open(self.LLM_call_logpath, 'a') as wf:
                for i in range(written, calls_made - 1):
                    wf.write(json.dumps(last_dist) + "\n")
                    print(f"[!] Writing a line to LLM call pth!, index {i}")
                # write the current distribution on the current call line
                wf.write(json.dumps(curr_dist) + "\n")
                print(f"[!] Writing a line to LLM call pth!")


        with open(self.log_pth, "a") as log_file:
            log_file.write("[main sampling] All samples done; starting backtracking.\n")
        self.recursive_backtrack(root)
        final_dist = root.target.prob
        with open(self.log_pth, "a") as log_file:
            log_file.write(f"[final prob] {json.dumps(final_dist)}\n")
        print(root)

    def sampling_to_leaf(self, allow_human=False):
        """
        One pass from root to leaf; mark every visited node as active.
        """
        node = self.tree.root
        node.active = True            # mark root
        parent = None
        while True:
            next_node, uncertainty = self.sampling_at_node(node, allow_human=allow_human)
            # mark this child too
            next_node.active = True
            if uncertainty is not None and parent is not None:
                idx = parent.children.index(node)
                parent.sigma[idx] = uncertainty
            if next_node is node:
                break
            parent = node
            node = next_node
        return

    def sampling_at_node(self, node, allow_human=False):
        """
        Handle sampling logic at a single node:
        - If it already has children, pick next via p_sigma and increment n.
        - Otherwise, run decision_S until we either spawn children or stop.
        Returns: (next_node, uncertainty) where uncertainty is None except on stop.
        """
        print(f"sampling at node {node.name}")
        # If node has existing children, sample among them via p_sigma
        if getattr(node, 'children', None):
            if node.n and node.p and node.sigma:
                self.sigma_update()
                child = self.p_sigma(node)
                idx = node.children.index(child)
                node.n[idx] += 1
                return child, None

        # Leaf: query the model until we spawn new children or stop
        while True:
            num, var_str = self.decision_S(node, allow_human)
            print(f"made decision {num}; ")
            # Propose variable (action 1)
            if num == 1:
                self.propose_variable_S(node)
            # Propose domain (action 2)
            elif num == 2:
                input_var = next((v for v in node.unbound if v.name == var_str), None)
                if input_var:
                    self.propose_domain_S(node, input_var)
            # Craft distribution and spawn children (action 3)
            elif num == 3:
                input_var = next((v for v in node.unbound if v.name == var_str), None)
                if input_var:
                    self.craft_distribution_S(node, input_var)
                    children = self.tree.spawn_children(node)
                    # choose next branch
                    self.sigma_update()
                    child = self.p_sigma(node)
                    idx = children.index(child)
                    node.n[idx] += 1
                    return child, None
            # Stop and craft target (action 4)
            elif num == 4:
                node, uncertainty = self.craft_target_distribution_S_uncertain(node)
                return node, uncertainty
            else:
                continue

    def nl_agent_entry_sampling(self, num_samples=100, allow_human=False, LLM_budget=10000, LLM_budget_logpth="./1.txt"):
        """
        Build the NL tree by sampling num_samples root-to-leaf paths.
        """
        self.LLM_budget = LLM_budget
        self.LLM_budget_left = LLM_budget
        self.LLM_call_logpath = LLM_budget_logpth
        
        root = self.tree.root
        with open(self.log_pth, "a") as log_file:
            log_file.write(f"[main NL sampling] Starting NL sampling with {num_samples} samples.\n")
        
        # Initial target distribution if budget allows
        if self.LLM_budget != 10000:
            self.craft_target_distribution_NL_uncertain(root)
            with open(self.LLM_call_logpath, 'w') as wf:
                wf.write(json.dumps(root.target.prob) + "\n")
            root.target.prob = None
        
        for i in range(1, num_samples + 1):
            if not self.LLM_budget_left > 0:
                break
            
            with open(self.log_pth, "a") as log_file:
                log_file.write(f"[main NL sampling {i}] Starting sample {i}\n")
            
            # Reset active flags
            self.tree.clear_active()
            
            # Sample to leaf
            self.sampling_to_leaf_NL(allow_human=allow_human)
            
            # Backtrack and update
            self.recursive_backtrack(root)
            
            print(f"LLM calls left: {self.LLM_budget_left}")
            # Log progress
            calls_made = self.LLM_budget - self.LLM_budget_left
            
            # Update LLM call log
            try:
                with open(self.LLM_call_logpath, 'r') as f:
                    existing = [json.loads(line) for line in f]
            except FileNotFoundError:
                existing = []
            
            written = len(existing)
            last_dist = existing[-1] if existing else None
            curr_dist = root.target.prob
            
            with open(self.LLM_call_logpath, 'a') as wf:
                for j in range(written, calls_made - 1):
                    wf.write(json.dumps(last_dist) + "\n")
                wf.write(json.dumps(curr_dist) + "\n")
        
        # Final backtrack
        self.recursive_backtrack(root)
        with open(self.log_pth, "a") as log_file:
            log_file.write(f"[final NL prob] {json.dumps(root.target.prob)}\n")

    def sampling_to_leaf_NL(self, allow_human=False):
        """
        One pass from root to leaf for NL tree; mark every visited node as active.
        """
        node = self.tree.root
        node.active = True
        parent = None
        
        while True:
            next_node, uncertainty = self.sampling_at_node_NL(node, allow_human=allow_human)
            next_node.active = True
            
            if uncertainty is not None and parent is not None:
                idx = parent.children.index(node)
                parent.sigma[idx] = uncertainty
            
            if next_node is node:
                break
            
            parent = node
            node = next_node

    def sampling_at_node_NL(self, node, allow_human=False):
        """
        Handle sampling logic at a single NL node.
        """
        with open(self.log_pth, "a") as log_file:
            log_file.write(f"Meta debug: sampling at node: {node}")
        
        # If node has existing children, sample among them
        if getattr(node, 'children', None):
            with open(self.log_pth, "a") as log_file:
                log_file.write(f"Meta debug: node already has children. Doing sampling.")
            
            self.sigma_update()
            child = self.p_sigma(node)
            idx = node.children.index(child)
            node.n[idx] += 1
            return child, None
        
        with open(self.log_pth, "a") as log_file:
            log_file.write(f"Meta debug: Node do not have children, processing.")
        # Leaf: query model until we spawn children or stop
        while True:
            num, var_str = self.decision_NL(node, allow_human)
            
            # Propose question (action 1)
            if num == 1:
                print("Debug: decided to propose another question")
                self.propose_question_NL(node)
            
            elif num == 2:
                print(f"Debug: decided to work on question '{var_str}'")
                
                # Find the specified question
                input_question = next((q for q in node.questions if q.name == var_str), None)
                if not input_question:
                    print(f"Exact match not found for '{var_str}', trying enhanced matching.")
                    
                    # Check if var_str contains brackets (indicating LLM included domain)
                    if '[' in var_str or ']' in var_str:
                        print("Variable string contains brackets, looking for base question name...")
                        
                        # Try finding questions whose names are contained in var_str
                        contained_matches = []
                        for q in node.questions:
                            if q.name in var_str:
                                contained_matches.append(q)
                                print(f"Found contained match: '{q.name[:50]}...' in '{var_str[:50]}...'")
                        
                        if len(contained_matches) == 1:
                            input_question = contained_matches[0]
                            print(f"✅ Using contained match: '{input_question.name[:50]}...'")
                        elif len(contained_matches) > 1:
                            # Use the longest match (most specific)
                            input_question = max(contained_matches, key=lambda q: len(q.name))
                            print(f"✅ Multiple matches found, using longest: '{input_question.name[:50]}...'")
                        else:
                            print("No contained matches found, trying reverse containment...")
                            
                            # Try finding if var_str is contained in any question name  
                            # (for cases where LLM truncated the question)
                            reverse_matches = []
                            clean_var_str = var_str.replace('[', '').replace(']', '').strip()
                            for q in node.questions:
                                if clean_var_str in q.name:
                                    reverse_matches.append(q)
                                    print(f"Found reverse match: '{clean_var_str}' in '{q.name[:50]}...'")
                            
                            if len(reverse_matches) == 1:
                                input_question = reverse_matches[0]
                                print(f"✅ Using reverse match: '{input_question.name[:50]}...'")
                            elif len(reverse_matches) > 1:
                                input_question = reverse_matches[0]  # Use first match
                                print(f"✅ Multiple reverse matches, using first: '{input_question.name[:50]}...'")
                            else:
                                print("ERROR: no matched question found even with robust matching.")
                
                print(f"Found question: '{input_question.name[:50]}...'")
                print(f"Current state - Domain: {bool(input_question.value)}, Probabilities: {bool(input_question.prob)}")
                
                has_domain = bool(input_question.value)
                has_probabilities = bool(input_question.prob)
                
                if has_domain and has_probabilities:
                    print("Question already fully defined! Ready to spawn children.")
                    # Spawn children immediately
                    children = self.tree.spawn_children(node, self)
                    if children:
                        print(f"SUCCESS: Spawned {len(children)} children")
                        self.sigma_update()
                        child = self.p_sigma(node)
                        idx = children.index(child)
                        node.n[idx] += 1
                        return child, None
                    else:
                        print("ERROR: Failed to spawn children despite question being ready!")
                        continue
                        
                elif has_domain and not has_probabilities:
                    print("Question has domain, needs probabilities")
                    # Only craft distribution
                    self.craft_distribution_NL(node, input_question)
                    
                    if input_question.prob:
                        print("Successfully set probabilities, now spawning children")
                        children = self.tree.spawn_children(node, self)
                        if children:
                            print(f"SUCCESS: Spawned {len(children)} children")
                            self.sigma_update()
                            child = self.p_sigma(node)
                            idx = children.index(child)
                            node.n[idx] += 1
                            return child, None
                        else:
                            print("ERROR: Failed to spawn children after setting probabilities!")
                            continue
                    else:
                        print("ERROR: Failed to set probabilities!")
                        continue
                        
                elif not has_domain:
                    print("Question needs domain first")
                    # Propose domain first
                    self.propose_domain_NL(node, input_question)
                    
                    if input_question.value:
                        print("Successfully set domain, now setting probabilities")
                        # Now craft distribution
                        self.craft_distribution_NL(node, input_question)
                        
                        if input_question.prob:
                            print("Successfully set probabilities, now spawning children")
                            children = self.tree.spawn_children(node, self)
                            if children:
                                print(f"SUCCESS: Spawned {len(children)} children")
                                self.sigma_update()
                                child = self.p_sigma(node)
                                idx = children.index(child)
                                node.n[idx] += 1
                                return child, None
                            else:
                                print("ERROR: Failed to spawn children!")
                                continue
                        else:
                            print("ERROR: Failed to set probabilities after setting domain!")
                            continue
                    else:
                        print("ERROR: Failed to set domain!")
                        continue
            
            # Stop and craft target (action 4)
            elif num == 3:
                print("Debug: decided to stop")
                node, uncertainty = self.craft_target_distribution_NL_uncertain(node)
                return node, uncertainty
            
            else:
                continue

    def p_sigma(self, node):
        """
        Choose a child index i that minimizes ||(n + e_i) - (p * sigma)||^2.
        Return that child.
        """
        target = [node.p[i] * node.sigma[i] for i in range(len(node.p))]
        best_idx = None
        best_dist = None
        for i in range(len(node.n)):
            new_n = node.n.copy()
            new_n[i] += 1
            dist = sum((new_n[j] - target[j])**2 for j in range(len(new_n)))
            if best_dist is None or dist < best_dist:
                best_dist = dist
                best_idx = i
        print("best index: ", best_idx)
        return node.children[best_idx]
    
    def sigma_update(self):
        """
        Kick off a full bottom-up pass to recompute every node’s sigma vector
        from the root of the tree.
        """
        self._sigma_recursion(self.tree.root)

    def _sigma_recursion(self, node):
        """
        Recurse into each child first; then, if this node has children,
        update its sigma vector entries for those children which themselves
        have children (i.e. non-leaves).  Leaf-child entries are left as is.
        """
        # Base case: nothing to do at leaves
        if not node.children:
            return

        # First, recurse down any non-leaf children
        for child in node.children:
            if child.children:
                self._sigma_recursion(child)

        # Now update this node’s sigma array
        # node.p is the probability distribution over its children
        # node.sigma is the current vector of per-child uncertainties
        for i, child in enumerate(node.children):
            # only recompute sigma[i] if this child has its own children
            if child.children:
                # weighted RMS of that child’s sigma vector
                # sigma_child[j] is the jth uncertainty under `child`
                ss = 0.0
                for j, σ in enumerate(child.sigma):
                    # child.p[j] is the probability of that grand-child branch
                    ss += child.p[j] * (σ ** 2)
                node.sigma[i] = math.sqrt(ss)
            # else: pure leaf, leave node.sigma[i] at its existing value
       
    # Take a json styled role-name array and return the generated content, supports placeholder replacements
    def single_LLM_call(self, json_prompt, client, name, replacements={}, additional_message=None):
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
        
        #with open(self.log_pth, "a") as log_file:
            #log_file.write(f"[log] Finished an LLM call. The response:\n{response.choices[0].message.content}\n\n")

        self.LLM_budget_left -= 1

        # Return the actual generation
        return response.choices[0].message.content
    

    # Take a json styled role-name array and return the generated content converted to a node object, supports placeholder replacements
    # do not revise the old node. Creates a new node with parent and children exactly the same as the old node and revise that node
    def single_LLM_call_to_node(self, json_prompt, client, name, replacements={}, S = True, old_node = None, additional_message=None):
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
        
        #with open(self.log_pth, "a") as log_file:
            #log_file.write(f"[log] Finished an LLM call. The response:\n{response.choices[0].message.content}\n\n")

        self.LLM_budget_left -= 1
        
        raw_json = response.choices[0].message.content

        #Extract the json, in case the LLM generates some explanation besides the json 
        start_idx = raw_json.find('{')
        end_idx = raw_json.rfind('}')
        if start_idx != -1 and end_idx != -1 and start_idx < end_idx:
            extracted_json = raw_json[start_idx:end_idx+1].strip()
        else:
            # If we can't find matching braces, just fall back to the entire string,
            # or handle the error some other way
            extracted_json = raw_json.strip()

        try:
            # If it’s truly Python dictionary-like, this should succeed:
            python_dict = ast.literal_eval(extracted_json)
            extracted_json = json.dumps(python_dict, ensure_ascii=False)
        except Exception as e:
            # As a fallback, try a naive replace of single quotes with double quotes
            # or just raise the error so you can see the unparsed content.
            with open(self.log_pth, "a") as log_file:
                log_file.write(f"[err] Could not parse LLM output with ast.literal_eval: {e}\n\n")
            # Fallback approach (be careful with embedded quotes):
            extracted_json = extracted_json.replace("'", '"')

        '''with open(self.log_pth, "a") as log_file:
            if S:
                log_file.write(f"[log] Attempting to convert the LLM call output to an S node.\n\n")
            else:
                log_file.write(f"[log] Attempting to convert the LLM call output to an NL node.\n\n")'''

        try:
            #dump to a temporary path
            with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as tmpfile:
                tmpfile.write(extracted_json)
                tmpfile.flush()
                temp_json_path = tmpfile.name

            # Now build the S_node from that temp file
            if S: 
                temp_node = S_node.from_json(temp_json_path, name=name)
            else:
                temp_node = NL_node.from_json(temp_json_path, name=name)

            '''with open(self.log_pth, "a") as log_file:
                log_file.write(f"[log] Convert to structured node successfully. the generated node:\n{temp_node.__repr__()}\n\n")'''
        
        except Exception as e:
            '''with open(self.log_pth, "a") as log_file:
                log_file.write(f"[err] Error parsing LLM JSON into S_node: {e}\n\n")'''
            return old_node 
        
        return temp_node

    # Edits a node by proposing a new variable for it, return the revised node
    def propose_variable_S(self, node):

        var_prompt = self.structured_prompts["var_prompt"]
        node_json_str = node.to_json()
        content = {"input": node_json_str}

        '''with open(self.log_pth, "a") as log_file:
            log_file.write(f"[log] Proposing a new variable for the node:\n{node_json_str}\n\n")'''
        
        temp_node = self.single_LLM_call_to_node(var_prompt, self.llm, node.name, content, True, node)
                          
        node.align_with(temp_node, False)


    # Edits a node by proposing domain for a variable in it, return the revised node
    def propose_domain_S(self, node, var):

        domain_prompt = self.structured_prompts["domain_prompt"]
        node_json_str = node.to_json()
        content = {
            "input": node_json_str,
            "var_name": var.name
        }

        '''with open(self.log_pth, "a") as log_file:
            log_file.write(f"[log] Proposing domain for the node:\n{node_json_str} \n and var {var.name}\n\n")'''
        
        temp_node = self.single_LLM_call_to_node(domain_prompt, self.llm, node.name, content, True, node)
        node.align_with(temp_node, False)



    # Edits a node by estimating the probability for a variable whose domain is already defined
    def craft_distribution_S(self, node, var):

        #confirm that the variable has its domain defined before we query for the prob distribution
        if not var.value or len(var.value) == 0:
            node = self.propose_domain_S(node, var)
            var = next((v for v in node.unbound if v.name == var.name), var)
        
        prob_prompt = self.structured_prompts["prob_prompt"]
        node_json_str = node.to_json()
        content = {
            "input": node_json_str,
            "var_name": var.name
        }

        '''with open(self.log_pth, "a") as log_file:
            log_file.write(f"[log] Proposing distribution for the node:\n{node_json_str} \n and var {var.name}\n\n")'''
        
        response = self.single_LLM_call(prob_prompt, self.llm, node.name, content)
        
        #parse result to be a list of probabilities
        float_candidates = re.findall(r"[-+]?\d*\.\d+|\d+", response)
        parsed_probabilities = [float(x) for x in float_candidates]


        #sanity checks
        if len(parsed_probabilities) != len(var.value):
            err_msg = (f"[err] The LLM returned {len(parsed_probabilities)} probabilities "
                    f"for variable '{var.name}', but there are {len(var.value)} domain values.")
            with open(self.log_pth, "a") as log_file:
                log_file.write(err_msg + "\n\n")
            
            return node
        total_prob = sum(parsed_probabilities)
        if not abs(total_prob - 1.0) < 1e-7:
            err_msg = (f"[err] The probabilities for variable '{var.name}' do not sum to 1. "
                    f"They sum to {total_prob:.3f}.")
            with open(self.log_pth, "a") as log_file:
                log_file.write(err_msg + "\n\n")

            return node

        #update var in node
        var.prob = parsed_probabilities
        '''with open(self.log_pth, "a") as log_file:
            log_file.write(f"[log] Updated {var.name} with probabilities {var.prob}\n\n")'''


        return node
    
    # Edits a node by estimating the probability of the target variable
    def craft_target_distribution_S(self, node):
        
        if not node.target.value or len(node.target.value) == 0:
            self.propose_domain_S(node, node.target)
            if not node.target.value or len(node.target.value) == 0:
                return node, None
            
        target_prob_prompt = self.structured_prompts["target_prob_prompt"]
        node_json_str = node.to_json()
        content = {
            "input": node_json_str,
        }

        with open(self.log_pth, "a") as log_file:
            log_file.write(f"[log] Crafting the target distribution and returning from node:\n{node_json_str} \n\n")
        
        prob_list = self.single_LLM_call(target_prob_prompt, self.llm, node.name, content)
        float_candidates = re.findall(r"[-+]?\d*\.\d+|\d+", prob_list)
        parsed_probabilities = [float(x) for x in float_candidates]

        #sanity checks
        if len(parsed_probabilities) != len(node.target.value):
            err_msg = (f"[err] The LLM returned {len(parsed_probabilities)} probabilities "
                    f"for variable '{node.target.name}', but there are {len(node.target.value)} domain values.")
            with open(self.log_pth, "a") as log_file:
                log_file.write(err_msg + "\n\n")
            
            return node
        total_prob = sum(parsed_probabilities)
        if not abs(total_prob - 1.0) < 1e-7:
            err_msg = (f"[err] The probabilities for variable '{node.target.name}' do not sum to 1. "
                    f"They sum to {total_prob:.3f}.")
            with open(self.log_pth, "a") as log_file:
                log_file.write(err_msg + "\n\n")

            return node, parsed_probabilities
        
        # Assign the parsed probabilities
        node.target.prob = parsed_probabilities
        return node, parsed_probabilities

    def craft_target_distribution_S_uncertain(self, node):
        """
        Parse LLM output of format "[p1, p2, ...], u" where u is uncertainty.
        """
        if not node.target.value or len(node.target.value) == 0:
            self.propose_domain_S(node, node.target)
            if not node.target.value:
                return node, None
        raw = self.single_LLM_call(
            self.structured_prompts["target_prob_prompt_uncertain"],
            self.llm, node.name,
            replacements={"input": node.to_json()}
        )
        # Extract list and uncertainty
        # match [0.2, 0.4, 0.4] and then comma and number
        m = re.match(r"\s*(\[.*?\])\s*,\s*([0-9.+-eE]+)", raw)
        if m:
            probs_str, unc_str = m.groups()
            try:
                probs = ast.literal_eval(probs_str)
                uncertainty = float(unc_str)
            except Exception:
                # fallback: parse floats
                float_vals = re.findall(r"[-+]?\d*\.\d+|\d+", probs_str)
                probs = [float(x) for x in float_vals]
                uncertainty = float(unc_str)
        else:
            # fallback: old behavior—only probs returned
            print("Did not parse target correctly.")
            float_vals = re.findall(r"[-+]?\d*\.\d+|\d+", raw)
            probs = [float(x) for x in float_vals[:len(node.target.value)]]
            # compute entropy as uncertainty
            uncertainty = -sum(p * math.log(p+1e-12) for p in probs)
        # assign and return
        node.target.prob = probs
        print(f"[!] Node {node.name} prob = {node.target.prob}")
        return node, uncertainty


    #function to query human about a variable at a node
    def human_question_S(self, node, var):
        node_str = node.__repr__()

        human_input = input(
            f"Please provide a distribution for variable '{var.name}' in the format "
            "'value: probability, value: probability, ...' "
            f"the current node is: {node_str}"
        ).strip()
        # Parse the human input
        domain_values = []
        probabilities = []
        try:
            parts = human_input.split(',')
            for part in parts:
                part = part.strip()
                if not part:
                    continue
                if ':' not in part:
                    raise ValueError("Each part must contain a ':' separating value and probability.")
                val, prob_str = part.split(':', 1)
                val = val.strip()
                prob = float(prob_str.strip())
                domain_values.append(val)
                probabilities.append(prob)
            # Verify that the probabilities sum to 1.
            total_prob = sum(probabilities)
            if not abs(total_prob - 1.0) < 1e-6:
                raise ValueError(f"Probabilities must sum to 1, but they sum to {total_prob}.")
        except Exception as e:
            with open(self.log_pth, "a") as log_file:
                log_file.write(f"Error parsing human input distribution: {str(e)}\n")
            return node
        
        var.value = domain_values
        var.prob = probabilities
        return node, var




    # Function to call the LLM to decide what step to take next. Returns the decision number and a text representing a variable to take action on, if any
    def decision_S(self, node, allow_human=False, additional_message=None):

        '''with open(self.log_pth, "a") as log_file:
            log_file.write(f"[main {node.name}] : calling the LLM to decide which step to take next. \n")'''
    
        #decides if allowing the LLM to choose to ask human for help
        if allow_human:
            des_prompt = self.structured_prompts["decision_prompt_h"]
        else:
            des_prompt = self.structured_prompts["decision_prompt"]


        #set up the messages
        node_json_str = node.to_json()
        importance = 1.25 - len(node.name)*0.25
        content = {
            "input": node_json_str,
            "imp": importance
        }

        #allows multiple tries if the result cannot parse what we need
        attempt = 0
        while attempt < 3:
            response = self.single_LLM_call(des_prompt, self.llm, node.name, replacements=content, additional_message=additional_message)
            match = re.search(r'^\s*(\d+)\s*(?:,\s*(.+))?\s*$', response)
            if match:
                decision_num = int(match.group(1))
                decision_text = match.group(2).strip() if match.group(2) else None

                '''if decision_text:
                    with open(self.log_pth, "a") as log_file:
                        log_file.write(f"[log] successfully called the LLM for decision. Returned decision {decision_num} and variable {decision_text} \n\n")
                else:
                    with open(self.log_pth, "a") as log_file:
                        log_file.write(f"[log] successfully called the LLM for decision. Returned decision {decision_num} \n\n")'''

                return decision_num, decision_text
            attempt += 1
        
        return None
    
    def nl_agent_entry(self, step = 100, allow_human = False):
        pass

    def propose_question_NL(self, node):
        """
        Propose a new question for the NL node using var_prompt.
        """
        var_prompt = self.natural_prompts["var_prompt"]
        
        # Format content using existing helper functions
        target_q = self.format_variables_for_prompt(node.target)
        sub_q = self.format_variables_for_prompt(node.questions)
        f = self.format_facts_for_prompt(node.facts)
        
        content = {
            "target_q": target_q,
            "sub_q": sub_q,
            "f": f
        }
        
        response = self.single_LLM_call(var_prompt, self.llm, node.name, content)
        
        # The response should be a new question text
        new_question = response.strip()
        print(f"Debug: new question: {new_question}")

        # Create new Variable for the question
        new_question_var = Variable(name=new_question, type="question", value=[], prob=None)
        
        # Add to node's questions
        node.questions.append(new_question_var)
        print(f"After proposing new question, the node is: {node}")
        
        with open(self.log_pth, "a") as log_file:
            log_file.write(f"[main {node.name}] proposed question: {new_question}\n")

    def propose_domain_NL(self, node, question_var):
        """
        Propose domain for a question variable in NL node using domain_prompt.
        """
        domain_prompt = self.natural_prompts["domain_prompt"]
        
        print(f"current variable, before proposing domain: {question_var}")
        # Format the specific question and facts for the domain prompt
        content = {
            "q": self.format_variables_for_prompt(question_var),
            "f": self.format_facts_for_prompt(node.facts)
        }
        
        response = self.single_LLM_call(domain_prompt, self.llm, node.name, content)
        
        # Parse domain values from response (expecting format like [value1; value2; value3])
        try:
            start_idx = response.find('[')
            end_idx = response.rfind(']')
            if start_idx != -1 and end_idx != -1:
                domain_str = response[start_idx+1:end_idx]
                # Split by semicolon and clean up
                domain_values = [val.strip() for val in domain_str.split(';') if val.strip()]
                print(f"Debug: domain values: {domain_values}")
                question_var.value = domain_values
            else:
                raise ValueError("Could not find domain list in response")
                
        except Exception as e:
            with open(self.log_pth, "a") as log_file:
                log_file.write(f"[err] Could not parse domain response: {e}\n")

    def craft_distribution_NL(self, node, question_var):
        """
        Craft probability distribution for a question variable in NL node.
        """
        # Ensure domain is defined first
        if len(question_var.value) == 0:
            self.propose_domain_NL(node, question_var)
        
        if not question_var.value:
            return
        
        prob_prompt = self.natural_prompts["prob_prompt"]
        
        content = {
            "q": self.format_variables_for_prompt(question_var),
            "f": self.format_facts_for_prompt(node.facts)
        }
        
        response = self.single_LLM_call(prob_prompt, self.llm, node.name, content)
        
        # Parse probabilities (expecting format like [0.2; 0.3; 0.5])
        try:
            start_idx = response.find('[')
            end_idx = response.rfind(']')
            if start_idx != -1 and end_idx != -1:
                prob_str = response[start_idx+1:end_idx]
                # Split by semicolon and convert to floats
                parsed_probabilities = [float(p.strip()) for p in prob_str.split(';') if p.strip()]
                print(f"Debug: parsed probabilities: {parsed_probabilities}")
            else:
                # Fallback: find all float numbers
                float_candidates = re.findall(r"[-+]?\d*\.\d+|\d+", response)
                parsed_probabilities = [float(x) for x in float_candidates]
                print(f"Debug: ERROR, did not parse prob correctly: {parsed_probabilities}")
        except Exception as e:
            with open(self.log_pth, "a") as log_file:
                log_file.write(f"[err] Could not parse probability response: {e}\n")
            return
        
        # Sanity checks
        if len(parsed_probabilities) != len(question_var.value):
            with open(self.log_pth, "a") as log_file:
                log_file.write(f"[err] Probability count mismatch for {question_var.name}\n")
            return
        
        total_prob = sum(parsed_probabilities)
        if not abs(total_prob - 1.0) < 1e-7:
            with open(self.log_pth, "a") as log_file:
                log_file.write(f"[err] Probabilities don't sum to 1 for {question_var.name}\n")
            return
        
        question_var.prob = parsed_probabilities

    def craft_target_distribution_NL(self, node):
        """
        Craft target distribution for NL node.
        """
        if not node.target.value or len(node.target.value) == 0:
            self.propose_domain_NL(node, node.target)
            if not node.target.value:
                print("error: target variable failed to define domain")
                return node
        
        target_prob_prompt = self.natural_prompts["prob_prompt"]
        
        # Format target question using the helper function
        content = {
            "q": self.format_variables_for_prompt(node.target),
            "f": self.format_facts_for_prompt(node.facts)
        }
        
        response = self.single_LLM_call(target_prob_prompt, self.llm, node.name, content)
        
        # Parse probabilities (expecting format like [0.2; 0.3; 0.5])
        try:
            start_idx = response.find('[')
            end_idx = response.rfind(']')
            if start_idx != -1 and end_idx != -1:
                prob_str = response[start_idx+1:end_idx]
                parsed_probabilities = [float(p.strip()) for p in prob_str.split(';') if p.strip()]
                print(f"Debug: parsed probabilities: {parsed_probabilities}")
            else:
                # Fallback: find all float numbers
                float_candidates = re.findall(r"[-+]?\d*\.\d+|\d+", response)
                parsed_probabilities = [float(x) for x in float_candidates]
                print(f"Debug: ERROR, did not parse prob correctly: {parsed_probabilities}")
        except Exception as e:
            with open(self.log_pth, "a") as log_file:
                log_file.write(f"[err] Could not parse target probability response: {e}\n")
            return node
        
        # Sanity checks
        if len(parsed_probabilities) != len(node.target.value):
            with open(self.log_pth, "a") as log_file:
                log_file.write(f"[err] Target probability count mismatch\n")
            return node
        
        total_prob = sum(parsed_probabilities)
        if not abs(total_prob - 1.0) < 1e-7:
            with open(self.log_pth, "a") as log_file:
                log_file.write(f"[err] Target probabilities don't sum to 1\n")
            return node
        
        node.target.prob = parsed_probabilities
        return node

    def craft_target_distribution_NL_uncertain(self, node):
        """
        Craft target distribution with uncertainty for NL node.
        """
        if not node.target.value or len(node.target.value) == 0:
            self.propose_domain_NL(node, node.target)
            if not node.target.value:
                return node, None
        
        target_prob_prompt = self.natural_prompts["target_prob_prompt_uncertain"]
        
        # Format target question using the helper function
        content = {
            "q": self.format_variables_for_prompt(node.target),
            "f": self.format_facts_for_prompt(node.facts)
        }
        
        response = self.single_LLM_call(target_prob_prompt, self.llm, node.name, content)
        
        # Parse format "[p1; p2; ...], uncertainty"
        try:
            # Look for pattern like [0.2; 0.3; 0.5], 0.8
            bracket_end = response.rfind(']')
            if bracket_end != -1:
                comma_pos = response.find(',', bracket_end)
                if comma_pos != -1:
                    # Extract probabilities part
                    prob_part = response[:bracket_end+1]
                    start_idx = prob_part.find('[')
                    if start_idx != -1:
                        prob_str = prob_part[start_idx+1:bracket_end]
                        probs = [float(p.strip()) for p in prob_str.split(';') if p.strip()]
                    else:
                        raise ValueError("Could not find probability list")
                    
                    # Extract uncertainty part
                    unc_part = response[comma_pos+1:].strip()
                    uncertainty = float(unc_part)
                else:
                    raise ValueError("Could not find comma separator")
            else:
                raise ValueError("Could not find closing bracket")
                
        except Exception as e:
            with open(self.log_pth, "a") as log_file:
                log_file.write(f"[err] Could not parse uncertain target response: {e}\n")
            # Fallback: parse only probabilities and compute entropy as uncertainty
            float_vals = re.findall(r"[-+]?\d*\.\d+|\d+", response)
            probs = [float(x) for x in float_vals[:len(node.target.value)]]
            uncertainty = -sum(p * math.log(p+1e-12) for p in probs)
        
        node.target.prob = probs
        return node, uncertainty

    def decision_NL(self, node, allow_human=False, additional_message=None):
        """
        Make decision for next action in NL node.
        """
        if allow_human:
            des_prompt = self.natural_prompts["decision_prompt_h"]
        else:
            des_prompt = self.natural_prompts["decision_prompt"]
        
        importance = 1.25 - len(node.name) * 0.25
        # Ensure importance stays within [0, 1] range
        importance = max(0.0, min(1.0, importance))

        # Format content using existing helper functions
        target_q = self.format_variables_for_prompt(node.target)
        sub_q = self.format_variables_for_prompt(node.questions)
        f = self.format_facts_for_prompt(node.facts)
        
        content = {
            "target_q": target_q,
            "sub_q": sub_q,
            "f": f,
            "imp": importance
        }
        
        # Make multiple attempts to get valid decision
        attempt = 0
        while attempt < 3:
            response = self.single_LLM_call(
                des_prompt, 
                self.llm, 
                node.name, 
                replacements=content, 
                additional_message=additional_message
            )
            
            # Parse decision number and optional variable name
            match = re.search(r'^\s*(\d+)\s*(?:,\s*(.+))?\s*$', response)
            if match:
                decision_num = int(match.group(1))
                decision_text = match.group(2).strip() if match.group(2) else None
                print(f"Decision made: {decision_num, decision_text}")
                if decision_num == 2:
                    print(f"Debug: node status for decision 2: {node}")
                #TODO: check if decision_text is among the question names. If not, then it is probablly that the llm also returns the domain. Then, look through the question names to see if there's one that is contained in the decision text (not the exact same, but part of.), if so, use that as the decision text. 
                return decision_num, decision_text
            
            attempt += 1
        
        return None, None

    def recursive_backtrack(self, node):
        # base case: leaf
        if not node.children:
            print(f"Leaf {node.name} prob = {node.target.prob}")
            return

        # Handle different node types - S_node vs NL_node
        target_var = None
        
        if isinstance(node, S_node):
            # For structured nodes: figure out which unbound variable was used to split into children
            for vp in node.unbound:
                # child.bound holds exactly the value picked at that branch
                if any(vp.name == bc.name for bc in node.children[0].bound):
                    target_var = vp
                    break
        elif isinstance(node, NL_node):
            # For NL nodes: find which question variable was used to split into children
            # The splitting variable should be one that has prob defined but is not in children's questions
            for question in node.questions:
                if question.prob is not None:
                    # Check if this question was used for splitting by seeing if it's missing from children
                    child_question_names = {q.name for q in node.children[0].questions} if node.children else set()
                    if question.name not in child_question_names:
                        target_var = question
                        break
        
        # get the original split-weights for each child
        if target_var and target_var.prob:
            original_weights = target_var.prob
        else:
            # if for some reason there's no splitting var, assume equal weights
            print("error: no splitting var or prob provided")
            original_weights = [1.0] * len(node.children)

        # Identify visited and unvisited children
        visited = []
        unvisited = []
        for i, child in enumerate(node.children):
            if getattr(child, "children", None):
                visited.append(i)
                print(f"Node {child.name} is visited because it has children.")
            elif child.target.prob is not None:
                visited.append(i)
                print(f"Node {child.name} is visited because it has a defined prob.")
                print(f"Node detail: {child}")
            else:
                unvisited.append(i)
                print(f"Node {child.name} is unvisited - will use root estimate.")

        # if we skipped *all* children, leave this node's prob alone
        if not visited and not unvisited:
            print(f"Node {node.name} had no children; leaving target.prob as is.")
            print(f"current node prob distribution: {node.target.prob}")
            return

        # Calculate total weight including both visited and unvisited nodes
        total_weight_all = sum(original_weights)
        
        # Accumulate weighted average including estimates for unvisited nodes
        accumulated = None
        
        # Process visited children (recurse first, then incorporate)
        for idx in visited:
            child = node.children[idx]
            weight = original_weights[idx] / total_weight_all
            
            # Recurse down first
            self.recursive_backtrack(child)
            
            # Then incorporate its distribution
            if accumulated is None:
                accumulated = [p * weight for p in child.target.prob]
            else:
                for j in range(len(accumulated)):
                    accumulated[j] += child.target.prob[j] * weight
        
        # Process unvisited children (use root estimate)
        for idx in unvisited:
            weight = original_weights[idx] / total_weight_all
            
            # Use root estimation for unvisited nodes
            if hasattr(self, 'root_estimate') and self.root_estimate:
                if accumulated is None:
                    accumulated = [p * weight for p in self.root_estimate]
                else:
                    for j in range(len(accumulated)):
                        accumulated[j] += self.root_estimate[j] * weight
                print(f"Used root estimate for unvisited node {node.children[idx].name} with weight {weight}")
            else:
                print(f"Warning: No root estimate available for unvisited node {node.children[idx].name}")

        # Write back the properly normalized target distribution
        if accumulated:
            node.target.prob = accumulated
        else:
            print(f"Warning: No accumulated distribution computed for node {node.name}")

         
class Node(ABC):
    """
    Base class for nodes.
    Attributes:
      - name: a string representing the node name.
      - prob: an optional probability associated with the node.
      - children: list of child nodes.
    """
    def __init__(self, name: str, prob=None):
        self.name = name
        self.prob = prob
        self.children = []
        self.parent = None
    
    def __repr__(self):
        return f"Node(name={self.name!r}, prob={self.prob}, children={self.children})"

    @abstractmethod  
    def ready_to_spawn(self):
        pass

    @abstractmethod  
    def align_with(self, temp_node):
        pass



class NL_node(Node):
    """
    A Node has:
      - name (str)
      - target (str)     REQUIRED, exactly one
      - questions (list[str], optional)
      - facts   (list[str], optional)
      - prob
    """
    def __init__(
        self,
        name: str,
        target: str,           # Single required Variable
        questions=None,               # optional list of Variables
        facts=None,                 # optional list of Variables
        prob=None,
    ):
        super().__init__(name, prob)
        if not isinstance(target, Variable):
            raise ValueError("`target` must be a Variable.")
        self.target = target
        self.questions = questions if questions else []
        self.facts = facts if facts else []

    @classmethod
    def from_json(cls, json_path: str, name: str = "Node from JSON"):
        """
        Fixed version - facts are extracted as strings from JSON
        """
        with open(json_path, 'r') as f:
            data = json.load(f)

        # Process the target: exactly one variable object is expected.
        target_list = data.get("Target", [])
        if len(target_list) != 1:
            raise ValueError(f"JSON must contain exactly one 'Target' object. Found {len(target_list)}.")
        target_data = target_list[0]
        target_variable = Variable(
            name=target_data.get("Name", "Unnamed"),
            type="target",
            value=target_data.get("Value", []),
            prob=target_data.get("Prob", None)
        )

        # Process questions (if any) - these remain as Variables
        questions_data = data.get("Questions", [])
        questions_variables = []
        for item in questions_data:
            questions_variables.append(
                Variable(
                    name=item.get("Name", "Unnamed"),
                    type="question",
                    value=item.get("Value", []),
                    prob=item.get("Prob", [])
                )
            )
        
        # Process facts (if any) - FIXED: extract as strings only
        facts_data = data.get("Facts", [])
        facts_strings = []
        for item in facts_data:
            # Extract just the name/text of the fact
            fact_text = item.get("Name", "")
            if fact_text:
                facts_strings.append(fact_text)

        prob = data.get("prob", None)
        return cls(
            name=name, 
            target=target_variable, 
            questions=questions_variables, 
            facts=facts_strings,  # Pass as list of strings
            prob=prob
        )

    def to_json(self):
        """
        Convert NL_node to JSON format - facts are represented as simple name entries
        """
        def variable_to_dict(var):
            d = {
                "Name": var.name,
                "Value": var.value
            }
            if var.prob is not None:
                d["Prob"] = var.prob
            return d

        # Convert string facts to JSON format (just name field)
        facts_json = []
        for fact_str in self.facts:
            facts_json.append({
                "Name": fact_str,
                "Value": [],  # Facts don't have domains
                "Prob": None  # Facts don't have probabilities
            })

        data = {
            "prob": self.prob,
            "Target": [variable_to_dict(self.target)],
            "Questions": [variable_to_dict(q) for q in self.questions],
            "Facts": facts_json
        }
        return data

    def __repr__(self):
        return (
            f"Node(name={self.name!r},\n"
            f"     target={self.target},\n"
            f"     questions={self.questions},\n"
            f"     facts={self.facts}, \n)"
            f"     prob={self.prob})"
        )
    
    def ready_to_spawn(self):
        if not self.questions:
            return None
        else:
            for question in self.questions:
                if question.value and question.prob:
                    return question
            return None
        
    def align_with(self, temp_node):
        pass




class S_node(Node):
    """
    A S_Node has:
      - name (str)
      - target (Variable)     REQUIRED, exactly one
      - unbound (list[Variable], optional)
      - bound   (list[Variable], optional)
      - prob
    """
    def __init__(self, name: str, target: 'Variable', unbound=None, bound=None, prob=None):
        super().__init__(name, prob)
        if not isinstance(target, Variable):
            raise ValueError("`target` must be a Variable.")
        self.target = target
        self.unbound = unbound if unbound else []
        self.bound = bound if bound else []
        self.CE = None
        self.n = []       # counts per child
        self.p = []       # initial distribution
        self.sigma = []   # cumulative uncertainties
        self.active = False

    @classmethod
    def from_json(
        cls,
        json_path: str,
        name: str = "S_Node from JSON",
    ):
        """
        Read a JSON file, parse it, and return a Node. 
        Expects a structure like:
        {
          "prob" : 0.74,
          "Target": [
            { "Name": "123", "Value": ["123", "123"], "prob": [0.5, 0.5] }
          ],
          "Bound_cond": [
            { "Name": "123", "Value": ["123"] },
            { "Name": "123", "Value": ["123"] }
          ],
          "Unbound_cond": [
            { "Name": "123", "Value": ["123"] },
            { "Name": "123", "Value": ["123"] }
          ]
        }
        - "Target" must be a list of exactly one object.
        -  Conds can be zero or more objects.
        """

        #load json
        with open(json_path, 'r') as f:
            data = json.load(f)

        #get target (exactly one)
        target_list = data.get("Target", [])
        if len(target_list) != 1:
            raise ValueError(
                "JSON must contain exactly one 'Target' object. "
                f"Found {len(target_list)}."
            )
        target_item = target_list[0]
        target_var = Variable(
            name=target_item.get("Name", "Unnamed"),
            type="target",
            value=target_item.get("Value", []),
            prob=target_item.get("Prob", None)
        )

        # get bound cond, can be undefined
        Bound_cond = data.get("Bound_cond", [])
        Bound_vars = []
        for item in Bound_cond:
            Bound_vars.append(
                Variable(
                    name=item.get("Name", "Unnamed"),
                    type="bound_cond",
                    value=item.get("Value", []),
                    prob=item.get("Prob", None)
                )
            )

        # get unbound cond, can be undefined
        Unbound_cond = data.get("Unbound_cond", [])
        Unbound_vars = []
        for item in Unbound_cond:
            Unbound_vars.append(
                Variable(
                    name=item.get("Name", "Unnamed"),
                    type="unbound_cond",
                    value=item.get("Value", []),
                    prob=item.get("Prob", None)
                )
            )

        # get prob of this node, can be undefined
        prob = data.get("prob", None)

        # Return a new S_Node instance
        return cls(
            name=name,
            target=target_var,
            unbound=Unbound_vars,
            bound=Bound_vars,
            prob=prob
        )

    def to_json(self):
        def variable_to_dict(var):
            # Create a dictionary with the required fields.
            d = {
                "Name": var.name,
                "Value": var.value
            }
            # Include the probability if available.
            if var.prob is not None:
                d["Prob"] = var.prob
            return d

        # Build the complete dictionary according to the expected schema.
        data = {
            "prob": self.prob,
            "Target": [variable_to_dict(self.target)],  # Exactly one target.
            "Bound_cond": [variable_to_dict(var) for var in self.bound],
            "Unbound_cond": [variable_to_dict(var) for var in self.unbound]
        }
        return data


    def __repr__(self):
        return (
            f"S_Node(name={self.name!r} "
            f"     target={self.target}, "
            f"     unbound={self.unbound}, "
            f"     bound={self.bound})"
            f"     prob={self.prob})"
        )
    
    def ready_to_spawn(self):
        if not self.unbound:
            return None
        else:
            for var in self.unbound:
                if var.value and var.prob:
                    return var
            return None

    def align_with(self, temp_node, prob = True):

        self.target = temp_node.target
        self.bound = temp_node.bound
        self.unbound = temp_node.unbound

        if prob and temp_node.prob is not None:
            self.prob = temp_node.prob






class Variable:
    """
    Represents a variable with:
      - a name (string)
      - a list of possible values (strings)
      - an optional probability distribution over those values.
      - an optional type indicator, default as unbound_cond
    """
    def __init__(self, name: str, type="unbound_cond", value=None, prob=None):

        # Normalize 'value' to always be a list of strings.
        if value is None:
            value = []
        elif isinstance(value, str):
            value = [value]

        # If a probability array was provided
        if prob is not None:
            # If user passed a single float, convert to a list
            if isinstance(prob, (int, float)):
                prob = [float(prob)]
            
            if len(prob) == 1:
                if len(value) == 1:
                    pass
                else:
                    raise ValueError(
                        f"You provided a single probability for variable '{name}', but the domain has {len(value)} values."
                    )
            else:
                # Check length match
                if len(prob) != len(value):
                    raise ValueError(
                        "Number of probabilities does not match number of values. "
                        f"Got {len(prob)} probabilities but {len(value)} values."
                    )

                # Check that sum(prob) == 1 within a small tolerance
                total_prob = sum(prob)
                if not abs(total_prob - 1.0) < 1e-9:
                    raise ValueError(
                        f"Probabilities must sum to 1. Sum is {total_prob}."
                    )

        # Assign to instance
        self.name = name
        self.value = value
        self.prob = prob
        self.type = type

    def name(self):
        return self.name

    def __repr__(self):
        return (f"Variable(name={self.name!r}, "
                f"value={self.value!r}, "
                f"prob={self.prob!r})"
                f"type={self.type!r})")

class Tree(ABC):
    """
    Generic tree structure that contains a root node.
    """
    def __init__(self, root: Node):
        self.root = root

    @abstractmethod  
    def spawn_children(self, node, agent=None):
        pass


class NLTree(Tree):
    """
    Tree for natural language nodes.
    """
    def __init__(self, root: NL_node):
        if not isinstance(root, NL_node):
            raise ValueError("Root must be an instance of NLNode.")
        super().__init__(root)
    
    def clear_active(self):
        """ Recursively clear the .active flag on every node. """
        def _rec(node):
            node.active = False
            for c in node.children:
                _rec(c)
        _rec(self.root)

    def __repr__(self):
        # return JSON with active flags
        def node_to_dict(node):
            d = node.to_json()
            d['active'] = node.active          
            d['children'] = [node_to_dict(c) for c in node.children]
            return d
        return json.dumps(node_to_dict(self.root), ensure_ascii=False, indent=2)
    
    def spawn_children(self, node, agent = None):
        """
        Spawn children for NL node by branching on a ready question.
        Uses agent.convert_question_to_facts to get properly formatted declarative facts.
        
        Args:
            node: The NL_node to spawn children from
            agent: The AgentS1 instance (needed to access convert_question_to_facts)
        """

        if agent == None:
            print("agent cannot be none!")
            return
        ready_question = node.ready_to_spawn()
        if ready_question is None:
            print("Debug: Error: No question ready!")
            return []
        
        print(f"Debug: question to split upon: {ready_question}")
        
        # Convert the question Variable to declarative fact strings using the agent's helper function
        try:
            fact_strings = agent.convert_question_to_facts(ready_question)
            if len(fact_strings) != len(ready_question.value):
                with open(agent.log_pth, "a") as log_file:
                    log_file.write(f"[err] Fact count mismatch for question {ready_question.name}\n")
                return []
        except Exception as e:
            with open(agent.log_pth, "a") as log_file:
                log_file.write(f"[err] Could not convert question to facts: {e}\n")
            return []
        
        # Get remaining questions (excluding the one we're branching on)
        remaining_questions = [copy.deepcopy(q) for q in node.questions if q is not ready_question]
        
        print(f"Debug: remaining questions: {remaining_questions}")

        children = []
        for idx, (domain_val, fact_str) in enumerate(zip(ready_question.value, fact_strings)):
            # Clone target Variable
            child_target = deepcopy(node.target)
            child_target.prob = None
            
            # Create child node with the new fact added to facts list (facts are strings)
            child = NL_node(
                name=node.name + str(idx),
                target=child_target,
                questions=remaining_questions,
                facts=node.facts + [fact_str],  # Add the converted fact string
                prob=node.prob * ready_question.prob[idx] if node.prob else ready_question.prob[idx]
            )
            child.parent = node
            children.append(child)
        
        node.children = children
        
        # Initialize sampling statistics
        node.p = list(ready_question.prob)
        node.n = [0] * len(children)
        node.sigma = [1.0] * len(children)
        
        return children



class STree(Tree):
    def __init__(self, root: S_node):
        super().__init__(root)

    def clear_active(self):
        """ Recursively clear the .active flag on every node. """
        def _rec(node):
            node.active = False
            for c in node.children:
                _rec(c)
        _rec(self.root)

    def __repr__(self):
        # return JSON with active flags
        def node_to_dict(node):
            d = node.to_json()
            d['active'] = node.active           # <-- include it in JSON dump
            d['children'] = [node_to_dict(c) for c in node.children]
            return d
        return json.dumps(node_to_dict(self.root), ensure_ascii=False, indent=2)
    
    def spawn_children(self, node):
        unbound_var = node.ready_to_spawn()
        if unbound_var is None:
            return []
        remaining = [copy.deepcopy(q) for q in node.unbound if q is not unbound_var]
        children = []
        for idx, val in enumerate(unbound_var.value):
            # clone the target Variable so it isn't shared among siblings
            child_target = deepcopy(node.target)
            child_target.prob = None

            child = S_node(
                name=node.name + str(idx),
                target=child_target,
                unbound=remaining,
                bound=node.bound + [
                    Variable(name=unbound_var.name, type=unbound_var.type, value=[val])
                ],
                prob=node.prob * unbound_var.prob[idx]
            )
            child.parent = node
            children.append(child)
        node.children = children
        # initialize sampling stats on node
        node.p = list(unbound_var.prob)
        node.n = [0] * len(children)
        node.sigma = [1.0] * len(children)
        return children

def main():
    parser = argparse.ArgumentParser(
        description="Test AgentS1 sampling entrypoint with file-based input. Mode (structured/NL) determined by config file."
    )
    parser.add_argument(
        "config",
        type=Path,
        help="Path to the agent config JSON file."
    )
    parser.add_argument(
        "input_file",
        type=Path,
        help="Path to a file containing a single input line (the query)."
    )
    parser.add_argument(
        "--log",
        type=Path,
        default=Path("./agent_sampling.log"),
        help="Path to write the agent's log."
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=100,
        help="Number of sample paths to draw."
    )
    parser.add_argument(
        "--allow-human",
        action="store_true",
        help="Whether to allow human intervention."
    )
    parser.add_argument(
        "--budget",
        type=int,
        default=30,
        help="LLM budget (number of calls allowed)."
    )
    args = parser.parse_args()

    # Read the single line from the input file
    with open(args.input_file, "r") as f:
        line = f.readline().strip()
        if not line:
            raise ValueError(f"The input file {args.input_file} is empty.")

    try:
        # Instantiate the agent (this reads the config and determines the mode)
        agent = AgentS1(str(args.config), line, str(args.log))
        
        # Create LLM log path based on mode determined by config
        mode_suffix = "nl" if not agent.structured else "structured"
        llm_log_pth = Path(f"./sampling_test_{mode_suffix}.txt")
        llm_log_pth.parent.mkdir(parents=True, exist_ok=True)
        llm_log_pth.touch(exist_ok=True)

        print(f"Running in {'Natural Language' if not agent.structured else 'Structured'} mode (from config)")
        print(f"Input: {line}")
        print(f"Samples: {args.samples}")
        print(f"LLM Budget: {args.budget}")
        print(f"Log file: {args.log}")
        print(f"LLM call log: {llm_log_pth}")
        
        # Run the appropriate sampling method based on config
        if not agent.structured:  # Natural Language mode
            print("Starting Natural Language agent sampling...")
            agent.nl_agent_entry_sampling(
                num_samples=args.samples,
                allow_human=args.allow_human,
                LLM_budget=args.budget,
                LLM_budget_logpth=str(llm_log_pth)
            )
            
            # Output NL results
            print("\n" + "="*50)
            print("NATURAL LANGUAGE RESULTS")
            print("="*50)
            print("Final target distribution:")
            if hasattr(agent, 'nl_root') and agent.nl_root.target.prob:
                print(f"Target: {agent.nl_root.target.name}")
                print(f"Domain: {agent.nl_root.target.value}")
                print(f"Probabilities: {agent.nl_root.target.prob}")
            else:
                print("No target distribution available")
            
            print("\nFull NL tree JSON:")
            print(json.dumps(agent.tree.root.to_json(), indent=2, ensure_ascii=False))
            
        else:  # Structured mode
            print("Starting Structured agent sampling...")
            agent.s_agent_entry_sampling(
                num_samples=args.samples,
                allow_human=args.allow_human,
                LLM_budget=args.budget,
                LLM_budget_logpth=str(llm_log_pth)
            )
            
            # Output structured results
            print("\n" + "="*50)
            print("STRUCTURED RESULTS")
            print("="*50)
            print("Final target distribution:")
            if hasattr(agent, 's_root') and agent.s_root.target.prob:
                print(f"Target: {agent.s_root.target.name}")
                print(f"Domain: {agent.s_root.target.value}")
                print(f"Probabilities: {agent.s_root.target.prob}")
            else:
                print("No target distribution available")
            
            print("\nFull structured tree JSON:")
            print(json.dumps(agent.tree.root.to_json(), indent=2, ensure_ascii=False))

    except Exception as e:
        print(f"Error during execution: {e}")
        import traceback
        traceback.print_exc()
        return 1

    print(f"\nExecution completed. Check {args.log} for detailed logs.")
    return 0

if __name__ == "__main__":
    main()