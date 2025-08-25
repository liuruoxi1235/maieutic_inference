from pathlib import Path
import json
import dotenv
import json
import numpy as np
import openai
import tempfile
import re
import ast
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
import argparse
import json
import copy
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

        self.LLM_call_logpath = "./Full_tree_agent_answers.txt"

        #initialize openai api
        import os
        from dotenv import load_dotenv
        load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY")
        openai.api_key = api_key
        self.llm = openai  # Use the module directly

        if self.structured:
            self.s_root = self.init_full_root_from_txt(input)
            self.propose_domain_S(self.s_root, self.s_root.target)
            self.tree = STree(self.s_root)
            self.queue.append(self.s_root)
        else:
            self.tree = NLTree(self.nl_root)
            self.queue.append(self.nl_root)

        
        with open(self.log_pth, "a") as log_file:
            log_file.write(f"[main 0] [step 0] : Initialization finished. Now going to take steps after decisions. \n")
            log_file.write(f"[main 0] [content] : {json.dumps(self.s_root.to_json())} \n")
            print(f"[main 0] [content] : {json.dumps(self.s_root.to_json())} \n")

    

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

    def s_agent_entry(self, max_step = 100, allow_human = False, LLM_budget = 10000, LLM_budget_logpth = "./2.txt"):      
        root = self.tree.root

        self.LLM_budget = LLM_budget
        self.LLM_budget_left = LLM_budget
        self.LLM_call_logpath = LLM_budget_logpth

        if self.LLM_budget != 10000:
            self.craft_target_distribution_S(self.s_root)
            with open(self.LLM_call_logpath, 'w') as wf:
                wf.write(json.dumps(self.s_root.target.prob) + "\n")
            print(f"initial target distribution {self.s_root.target.prob} at root node {self.s_root}.")
            self.s_root.target.prob = None

        #we can either set a finite number of steps to take or let the model decide when to stop
        step_mode = True
        
        with open(self.log_pth, "a") as log_file:
            log_file.write(f"[main 0] [step 0] : Starting agentic workflow. Allowing {max_step} steps. \n")
            
        if step_mode:

            step = 0
            additional_msg = None
            stopped_node = []
            while(step < max_step and self.LLM_budget_left > 0):
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
                            with open(self.log_pth, "a") as log_file:
                                log_file.write(f"[err] Resolved. \n\n")
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

                    #Backtrack to get the answer (target variable value estimation at the root)
                    with open(self.log_pth, "a") as log_file:
                        log_file.write(f"[DEBUG before backtrack – leaves]\n")
                        for dist, p in self.get_leaves_distributions():
                            log_file.write(f"  leaf_dist={json.dumps(dist)}, branch_prob={p}\n")
                        log_file.write(f"\n root dist: {self.tree.root.target.prob}\n")
                    self.recursive_backtrack(root)
                    with open(self.log_pth, "a") as log_file:
                        log_file.write(f"[DEBUG after  backtrack – leaves]\n")
                        for dist, p in self.get_leaves_distributions():
                            log_file.write(f"  leaf_dist={json.dumps(dist)}, branch_prob={p}\n")
                        log_file.write(f"\n root dist: {self.tree.root.target.prob}\n")

                self.recursive_backtrack(root)
                # Compute number of LLM calls made so far
                calls_made = self.LLM_budget - self.LLM_budget_left
                print(f"Result after llm call {calls_made} with decision {num}: {self.tree.root.target.prob}")
                # Read existing lines
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
                    # write the current distribution on the current call line
                    if curr_dist is not None:
                        wf.write(json.dumps(curr_dist) + "\n")
                    else: 
                        wf.write(json.dumps(last_dist) + "\n")

            #after the given number of steps are taken, terminate all currently pending nodes, and trace backward
            while self.queue:
                node = self.queue.popleft()
                self.craft_target_distribution_S(node)
                with open(self.log_pth, "a") as log_file:
                    log_file.write(f"[main {node.name}] [GSS replacement internal steps] : Crafted target distribution at this node to start backtracking. \n")
                    log_file.write(f"[main {node.name}] [content] : {json.dumps(node.to_json())}\n")
            
            self.recursive_backtrack(root)


            print(root.__repr__())




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

        '''with open(self.log_pth, "a") as log_file:
            log_file.write(f"[log] Crafting the target distribution and returning from node:\n{node_json_str} \n\n")'''
        
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
    

    #These are left empty on purpuse. I want to implement the structured nodes first
    def nl_agent_entry(self, step = 100, allow_human = False):
        pass

    def propose_question_NL(self, node):
        pass

    def propose_domain_NL(self, node, var):
        pass

    def craft_distribution_NL(self, node, var):
        pass
    
    def human_question_NL(self, node, var):
        pass

    def decision_NL(self, node):
        pass

    def get_leaves_distributions(self):
        """
        Walk the full tree and collect, for each leaf:
          (leaf.target.prob, leaf.prob)
        where leaf.prob is the probability of reaching that leaf.
        Returns: List[ Tuple[List[float], float] ]
        """
        leaves = []
        def dfs(node):
            if not node.children:
                # node.target.prob is the final distribution at this leaf
                # node.prob is the probability of reaching it
                leaves.append((node.target.prob, node.prob))
            else:
                for c in node.children:
                    dfs(c)
        dfs(self.tree.root)
        return leaves
            
    
    # Recursively backtrack to get the distribution of the target variable at the root
    # Calculation is simple -- basically take an average of the children's target distributions, use the prob of the childrens as weight
    # Eg. if node has child 1 with prob 0.5, child 2 with prob 0.5, child 1 has target var distribution as ["rainy", "sunny"] with [0.2, 0.8], and child 1 has target var distribution as ["rainy", "sunny"] with [0.3, 0.7]
    # Then node should have target var distribution as ["rainy", "sunny"] with [0.25, 0.75]
    def recursive_backtrack(self, node):
        print("[!] Recursion starts!")
        # base case: leaf
        if not node.children:
            print(f"Leaf {node.name} prob = {node.target.prob}")
            return

        # figure out which unbound variable was used to split into these children
        target_var = None
        for vp in node.unbound:
            # child.bound holds exactly the value picked at that branch
            if any(vp.name == bc.name for bc in node.children[0].bound):
                target_var = vp
                break

        # get the original split-weights for each child
        if target_var and target_var.prob:
            original_weights = target_var.prob
        else:
            # if for some reason there's no splitting var, assume equal weights
            print("error: no splitting var or prob provided")
            original_weights = [1.0] * len(node.children)

        # collect only the indices of children that actually got visited
        visited = []
        for i, child in enumerate(node.children):
            if getattr(child, "children", None):
                visited.append(i)
                print(f"Node {child.name} is visited because it has children.")
            elif child.target.prob is not None:
                visited.append(i)
                print(f"Node {child.name} is visited because it has a defined prob.")
                print(f"Node detail: {child}")
            else:
                print(f"Skipping unvisited branch {child.name}")

        # if we skipped *all* children, leave this node's prob alone
        if not visited:
            print(f"Node {node.name} had no visited children; leaving target.prob as is.")
            if node.target.prob is None:
                self.craft_target_distribution_S(node)
            print(f"current node prob distribution: {node.target.prob}")
            return

        # renormalize the weights over just the visited branches
        total_w = sum(original_weights[i] for i in visited)
        norm_ws = [original_weights[i] / total_w for i in visited]

        # now accumulate a weighted average of the children's target distributions
        accumulated = None
        for w, idx in zip(norm_ws, visited):
            child = node.children[idx]
            # first recurse down
            self.recursive_backtrack(child)
            # then incorporate its distribution
            if accumulated is None:
                # start the vector
                accumulated = [p * w for p in child.target.prob]
            else:
                # add in weighted increments
                for j in range(len(accumulated)):
                    accumulated[j] += child.target.prob[j] * w

        # write back the properly normalized target distribution
        node.target.prob = accumulated
    
        print(f"Recursion ends with prob {node.target.prob}")


         
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
    def from_json(
        cls,
        json_path: str,
        name: str = "Node from JSON",
    ):
        """
        Read a JSON file, parse it, and return a Node. 
        Expects a structure like:
        {
          "prob" : 0.74,
          "Target": [
            { "Name": "123", "Value": [], "prob": [] }
          ],
          "Questions": [
            { "Name": "Are you happy?", "Value": [], "prob": [] },
            { "Name": "Is it raining?", "Value": ["Yes", "No"], "prob": [0.5, 0.5] }
          ],
          "Facts": [
            { "Name": "Is it snowing?", "Value": ["Yes"] }
          ]
        }
        - "Target" must be a list of exactly one variable object.
        -  questions and facts can be zero or more variable objects.
        """

        #load json
        with open(json_path, 'r') as f:
            data = json.load(f)

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

        # Process questions (if any)
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
        
        # Process facts (if any)
        facts_data = data.get("Facts", [])
        facts_variables = []
        for item in facts_data:
            facts_variables.append(
                Variable(
                    name=item.get("Name", "Unnamed"),
                    type="fact",
                    value=item.get("Value", []),
                    prob=item.get("Prob", [])
                )
            )

        prob = data.get("prob", None)
        return cls(name=name, target=target_variable, questions=questions_variables, facts=facts_variables, prob=prob)

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
    def spawn_children(self, node):
        pass


class NLTree(Tree):
    """
    Tree for natural language nodes.
    """
    def __init__(self, root: NL_node):
        if not isinstance(root, NL_node):
            raise ValueError("Root must be an instance of NLNode.")
        super().__init__(root)
    
    def spawn_children(self, node):
        question = node.ready_to_spawn()
        if question is None:
            return []
        

        children = []
        remaining_questions = [q for q in node.questions if q != question]
        for idx, answer in enumerate(question.value):

            # define name. eg: "341" -> "3411"
            child_name = node.name + str(idx)

            child_prob = node.prob * question.prob[idx]

            # Create a new fact
            new_fact = Variable(
                name=question.name,
                type=question.type,
                value=[answer],
                prob=[question.prob[idx]] if question.prob else None
            )
            child_facts = list(node.facts)
            child_facts.append(new_fact)

            child_node = NL_node(
                name=child_name,
                target=node.target,
                questions=remaining_questions,
                facts=child_facts,
                prob=child_prob
            )
            child_node.target.prob = None
            children.append(child_node)
            self.add_child_to_node(node.name, child_node)
        
        return children



class STree(Tree):
    """
    Tree for structured nodes.
    """
    def __init__(self, root: S_node):
        if not isinstance(root, S_node):
            raise ValueError("Root must be an instance of SNode.")
        super().__init__(root)
    
    def __repr__(self):
        # return JSON
        def node_to_dict(node):
            d = node.to_json()
            d['children'] = [node_to_dict(c) for c in node.children]
            return d
        return json.dumps(node_to_dict(self.root), ensure_ascii=False, indent=2)

    def spawn_children(self, node):
        unbound_var = node.ready_to_spawn()
        if unbound_var is None:
            return []

        # deep‐copy the remaining unbound vars so each branch has its own list
        remaining_unbound_vars = [
            copy.deepcopy(q)
            for q in node.unbound
            if q is not unbound_var
        ]

        for idx, value in enumerate(unbound_var.value):
            # define name. e.g. "341" -> "3411"
            child_name = node.name + str(idx)
            child_prob = node.prob * unbound_var.prob[idx]

            # create a fresh target copy for this branch
            child_target = copy.deepcopy(node.target)

            # create a new bound var for this branch
            new_bound_var = Variable(
                name=unbound_var.name,
                type=unbound_var.type,
                value=[value],
            )
            # deep‐copy the existing bound list then append the new one
            child_bound_vars = [copy.deepcopy(v) for v in node.bound] + [new_bound_var]

            child_node = S_node(
                name=child_name,
                target=child_target,
                unbound=remaining_unbound_vars,
                bound=child_bound_vars,
                prob=child_prob
            )
            node.children.append(child_node)
            child_node.parent = node

        return node.children

def main():
    parser = argparse.ArgumentParser(
        description="Test AgentS1 sampling entrypoint with file-based input."
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
    args = parser.parse_args()

    # Read the single line from the input file
    with open(args.input_file, "r") as f:
        line = f.readline().strip()
        if not line:
            raise ValueError(f"The input file {args.input_file} is empty.")
    
    log_pth = Path("/Users/ruoxiliu/Desktop/Argo_research/reasoning/sense/backtracking/experiments/compare_sampling_full/full_test_2.txt")
    log_pth.parent.mkdir(parents=True, exist_ok=True)
    log_pth.touch(exist_ok=True)

    # Instantiate and run sampling
    agent = AgentS1(str(args.config), line, str(args.log))
    agent.s_agent_entry(
        allow_human=args.allow_human,
        LLM_budget=20,
        LLM_budget_logpth=log_pth
    )

    # Output final results
    print("Final target distribution:")
    print(agent.s_root.target)
    print("\nFull tree JSON:")
    print(json.dumps(agent.tree.root.to_json(), indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()