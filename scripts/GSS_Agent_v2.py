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

dotenv.load_dotenv('.env')

class AgentS1():
    def __init__(self, config, log_pth='./example_files', ):
        with open(config, 'r') as config_file:
            schema = json.load(config_file)

        #storing if we are working with structured data or natural language data
        self.structured = schema["agent"]["structured"] 

        #model we prefer to use
        self.model_ID = schema["agent"]["model_dial"]

        #temperature
        self.temperature = schema["agent"]["temperature"]

        #the schema
        self.schema = schema["agent"]["structured_prompts"]["node_schema"]
        self.gss_df = pd.read_csv("1.csv")

        self.INVALID_RESPONSES = {"don't know", "iap", "not available in this year", "no answer", "skipped on web", "refused"}

        #logging
        log_pth = Path(schema["agent"]["log_pth"])
        log_pth.parent.mkdir(parents=True, exist_ok=True)
        log_pth.touch(exist_ok=True)
        self.log_pth = log_pth

        step_hist_pth = Path(schema["agent"]["step_hist"])
        step_hist_pth.parent.mkdir(parents=True, exist_ok=True)
        step_hist_pth.touch(exist_ok=True)
        self.step_hist_pth = step_hist_pth

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
                    log_file.write(f"nl root: {self.nl_root}")

        #initialize a queue for unprocessed nodes (leaves)
        self.queue = deque()

        #initialize openai api
        api_key = 'YOUR_API_KEY_HERE'  # Replace with your actual API key
        self.llm = openai.OpenAI(api_key=api_key)

        if self.structured:
            with open(self.log_pth, "a") as log_file:
                log_file.write(f"nl_root: {self.nl_root} \n")
            self.s_root = self.init_root_from_txt("/Users/ruoxiliu/Desktop/Argo_research/reasoning/sense/backtracking/vars_as_target.txt")
            self.tree = STree(self.s_root)
            self.queue.append(self.s_root)
        else:
            self.tree = NLTree(self.nl_root)
            self.queue.append(self.nl_root)

        
        with open(self.log_pth, "a") as log_file:
            log_file.write(f"[log] Finished initialization successfully. Now going to take steps after decisions. \n\n")

    

    def init_root_from_txt(self, file_path="1.txt"):
        with open(file_path, "r") as f:
            lines = f.readlines()
        if not lines:
            raise ValueError(f"{file_path} is empty!")
        chosen_line = random.choice(lines).strip()
        # Initialize a target Variable using the chosen line.
        # Instead of providing an empty list for prob, we pass None so that no probability validation is triggered.
        target_var = Variable(name=chosen_line, type="target", value=[], prob=None)
        # Create an S_node with:
        # - name set to "0"
        # - target set to the above Variable,
        # - empty unbound and bound lists,
        # - and a default probability of 1.
        root_node = S_node(name="0", target=target_var, unbound=[], bound=[], prob=1)

        label_description = self.find_best_matching_label(chosen_line , "/Users/ruoxiliu/Desktop/Argo_research/reasoning/sense/backtracking/GSS2022-varbook.txt")
        domain_dist, _ = self.get_variable_distribution(label_description[0])
        root_node.target.value = [category for category, pct in domain_dist]

        print("initialized root: ", root_node.__repr__())
        return root_node


    def s_agent_entry(self, max_step = 100, allow_human = False):      
        root = self.tree.root

        #we can either set a finite number of steps to take or let the model decide when to stop
        if max_step != -1:
            step_mode = True 
        
        with open(self.log_pth, "a") as log_file:
            log_file.write(f"[log] Starting agentic workflow. Allowing {max_step} steps. \n\n")
            
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
                    children = self.tree.spawnchildren(var)
                    for child in children:
                        self.queue.append(child)

                # let model decide which step to take next:         
                num, var_str = self.decision_S(node, allow_human, additional_msg)
                additional_msg = None

                # 1: propose a new variable
                if num == 1:

                    # this is the node with the variale proposed
                    self.propose_variable_S(node)

                    with open(self.step_hist_pth, "a") as log_file:
                        log_file.write(f"[step {step}] proposed variable (action 1)\n")
                        log_file.write(f"[step {step}] node: {json.dumps(node.to_json(), indent=2)}\n\n")
 
                #2: choose the domain for an unbound variable
                elif num == 2:
                    input_var = None
                    for var in node.unbound:
                        if var.name == var_str:
                            input_var = var

                    self.propose_domain_S(node, input_var)

                    with open(self.step_hist_pth, "a") as log_file:
                        log_file.write(f"[step {step}] chose domain (action 2)\n")
                        log_file.write(f"[step {step}] node: {json.dumps(node.to_json(), indent=2)}\n\n")

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

                    with open(self.step_hist_pth, "a") as log_file:
                        log_file.write(f"[step {step}] estimated distribution (action 3)\n")
                        log_file.write(f"[step {step}] node: {json.dumps(node.to_json(), indent=2)}\n\n")
                    
                
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
                            with open(self.step_hist_pth, "a") as log_file:
                                log_file.write(f"[step {step}] stopped at this node (action 4)\n")
                                log_file.write(f"[step {step}] node: {json.dumps(node.to_json(), indent=2)}\n\n")
            
            #after the given number of steps are taken, terminate all currently pending nodes, and trace backward
            while self.queue:
                node = self.queue.popleft()
                self.craft_target_distribution_S(node)
            
            self.recursive_backtrack(root)

            def compute_delta_CE(node, parent_ce=None):

                if parent_ce is not None:
                    delta_CE = parent_ce - node.CE
                    node_depth = len(node.name)
                    with open("/Users/ruoxiliu/Desktop/Argo_research/reasoning/sense/backtracking/target_deltaCE_depth.txt", "a") as ce_file:
                        ce_file.write(f"{node_depth}\t{delta_CE}\n")
                
                for child in node.children:
                    compute_delta_CE(child, parent_ce=node.CE)
            
            compute_delta_CE(self.s_root)

            print(root.__repr__())




    # Take a json styled role-name array and return the generated content, supports placeholder replacements
    def single_LLM_call(self, json_prompt, client, replacements={}, additional_message=None):
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
            log_file.write(f"[log] Doing an LLM call. The prompt messages:\n{messages}\n\n")

        # Call the model
        response = client.chat.completions.create(
            model=self.model_ID,
            messages=messages,
            temperature=self.temperature
        )
        
        with open(self.log_pth, "a") as log_file:
            log_file.write(f"[log] Finished an LLM call. The response:\n{response.choices[0].message.content}\n\n")

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
            log_file.write(f"[log] Doing an LLM call. The prompt messages:\n{messages}\n\n")

        # Call the model
        response = client.chat.completions.create(
            model=self.model_ID,
            messages=messages,
            temperature=self.temperature
        )
        
        with open(self.log_pth, "a") as log_file:
            log_file.write(f"[log] Finished an LLM call. The response:\n{response.choices[0].message.content}\n\n")

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

        with open(self.log_pth, "a") as log_file:
            if S:
                log_file.write(f"[log] Attempting to convert the LLM call output to an S node.\n\n")
            else:
                log_file.write(f"[log] Attempting to convert the LLM call output to an NL node.\n\n")

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

            with open(self.log_pth, "a") as log_file:
                log_file.write(f"[log] Convert to structured node successfully. the generated node:\n{temp_node.__repr__()}\n\n")
        
        except Exception as e:
            with open(self.log_pth, "a") as log_file:
                log_file.write(f"[err] Error parsing LLM JSON into S_node: {e}\n\n")
            return old_node 
        
        return temp_node

    # Edits a node by proposing a new variable for it, return the revised node
    def propose_variable_S(self, node):

        var_prompt = self.structured_prompts["var_prompt"]
        node_json_str = node.to_json()
        content = {"input": node_json_str}

        with open(self.log_pth, "a") as log_file:
            log_file.write(f"[log] Proposing a new variable for the node:\n{node_json_str}\n\n")

        GSS_found = 0

        additional_msg = None
        attempts = 0
        rejected_LLM_vars = set()
        while GSS_found == 0 and attempts < 8:
            attempts = attempts + 1
            #for the new node, we are essentially editing the node (by adding a var), so we would use the same name
            temp_node = self.single_LLM_call_to_node(var_prompt, self.llm, node.name, content, True, node, additional_msg)

            original_unbound_names = {v.name for v in node.unbound}
            new_vars = [v for v in temp_node.unbound if v.name not in original_unbound_names]
            if len(new_vars) != 1:
                print(f"error: proposed not exactly one new variable. Proposed {len(new_vars)}")
            new_var = new_vars[0]

            #print(f"debug1: LLM proposed variable {new_var.name}")


            label_description = self.find_best_matching_label(new_var.name , "/Users/ruoxiliu/Desktop/Argo_research/reasoning/sense/backtracking/GSS2022-varbook.txt")

            #If we did not find a match in the GSS dataset
            if label_description == "N/A":
                additional_msg = [
                    ("assistant", f"{temp_node.to_json()}"),
                    ("user", f"Your proposed variable is invalid. Rejected variables so far: {', '.join(sorted(rejected_LLM_vars))}. Try another variable that you have in mind and return in the same format. ")
                ]
                rejected_LLM_vars.add(new_var.name)
                #print(f"debug2: got an N/A from GSS")
                continue
            #If we found a match in the GSS dataset
            else:
                self.traversed_vars.add(label_description[1])

                for var in temp_node.unbound:
                    if var.name == new_var.name:
                        var.name = label_description[1]
                #The variable also needs to have a valid domain
                try:
                    domain_dist, _ = self.get_variable_distribution(label_description[0])
                except ValueError as e:
                    additional_msg = [
                        ("assistant", f"{temp_node.to_json()}"),
                        ("user", f"Your proposed variable is invalid. Rejected variables so far: {', '.join(sorted(rejected_LLM_vars))}. Try another variable that you have in mind and return in the same format. ")
                    ]
                    rejected_LLM_vars.add(new_var.name)
                    #print(f"debug3: got variable {label_description[1]} from GSS, but domain failed")
                    continue
                if len(domain_dist) > 7:
                    additional_msg = [
                        ("assistant", f"{temp_node.to_json()}"),
                        ("user", f"Your proposed variable is invalid. Rejected variables so far: {', '.join(sorted(rejected_LLM_vars))}. Try another variable that you have in mind and return in the same format. ")
                    ]
                    rejected_LLM_vars.add(new_var.name)
                    #print(f"debug3: got variable {label_description[1]} from GSS, but domain larger than 7")
                    print(f"domains: {domain_dist}")
                    continue

                #print(f"debug3: got variable {label_description[1]} from GSS, success!")
                for var in temp_node.unbound:
                    if var.name == label_description[1]:
                        var.value = [category for category, pct in domain_dist]

                #Now we can finally exit the while loop
                GSS_found = 1
                  
        node.align_with(temp_node, False)


    # Edits a node by proposing domain for a variable in it, return the revised node
    def propose_domain_S(self, node, var):

        domain_prompt = self.structured_prompts["domain_prompt"]
        node_json_str = node.to_json()
        content = {
            "input": node_json_str,
            "var_name": var.name
        }

        with open(self.log_pth, "a") as log_file:
            log_file.write(f"[log] Proposing domain for the node:\n{node_json_str} \n and var {var.name}\n\n")
        
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

        with open(self.log_pth, "a") as log_file:
            log_file.write(f"[log] Proposing distribution for the node:\n{node_json_str} \n and var {var.name}\n\n")
        
        response = self.single_LLM_call(prob_prompt, self.llm, content)
        
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
        with open(self.log_pth, "a") as log_file:
            log_file.write(f"[log] Updated {var.name} with probabilities {var.prob}\n\n")


        #Compute CE of branching factors
        try:
            constraints_desc = self.get_constraints(node)
            # For our evaluation, we assume the target description is node.target.name.
            # (In your setup, this should correspond to the varbook description.)
            ce_loss = self.compute_cross_entropy_loss(
                target_desc=var.name,
                predicted_distribution=parsed_probabilities,
                constraints_desc=constraints_desc,
                varbook_path="/Users/ruoxiliu/Desktop/Argo_research/reasoning/sense/backtracking/GSS2022-varbook.txt"
            )
            node_depth = len(node.name)
            # Write the (CE loss, depth) pair to file.
            ce_output_path = "/Users/ruoxiliu/Desktop/Argo_research/reasoning/sense/backtracking/branching_CE_depth.txt"
            with open(ce_output_path, "a") as ce_file:
                ce_file.write(f"{node_depth}\t{ce_loss}\n")
                node.CE = ce_loss
        except Exception as e:
            print(f"got error when deriving cross entropy: {e}")

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
        
        prob_list = self.single_LLM_call(target_prob_prompt, self.llm, content)
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

        with open(self.log_pth, "a") as log_file:
            log_file.write(f"[log] calling the LLM to decide which step to take next. \n\n")
    
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
            response = self.single_LLM_call(des_prompt, self.llm, replacements=content, additional_message=additional_message)
            match = re.search(r'^\s*(\d+)\s*(?:,\s*(.+))?\s*$', response)
            if match:
                decision_num = int(match.group(1))
                decision_text = match.group(2).strip() if match.group(2) else None

                if decision_text:
                    with open(self.log_pth, "a") as log_file:
                        log_file.write(f"[log] successfully called the LLM for decision. Returned decision {decision_num} and variable {decision_text} \n\n")
                else:
                    with open(self.log_pth, "a") as log_file:
                        log_file.write(f"[log] successfully called the LLM for decision. Returned decision {decision_num} \n\n")

                return decision_num, decision_text
            attempt += 1
        
        return None
    

    def find_best_matching_label(self, query: str, varbook_path: str, top_k: int = 20):
        with open(varbook_path, 'r') as f:
            lines = f.readlines()

        lhs_labels, rhs_labels = [], []
        for line in lines:
            line = line.strip()
            if ':' in line:
                lhs, rhs = line.split(':', 1)
                lhs_labels.append(lhs.strip())
                rhs_labels.append(rhs.strip())
            else:
                lhs_labels.append(line.strip())
                rhs_labels.append(line.strip())

        model = SentenceTransformer('roberta-base-nli-stsb-mean-tokens')
        rhs_embeddings = model.encode(rhs_labels, convert_to_numpy=True)

        query_embedding = model.encode([query], convert_to_numpy=True)

        faiss.normalize_L2(rhs_embeddings)
        faiss.normalize_L2(query_embedding)

        index = faiss.IndexFlatIP(rhs_embeddings.shape[1])
        index.add(rhs_embeddings)

        distances, indices = index.search(query_embedding, top_k)

        top_matches = [f"{i}. {rhs_labels[i]}" for i in indices[0] if i >= 0 and rhs_labels[i] not in self.traversed_vars]

        prompt = (
            f'In the following variables, find a single variable whose meaning is similar to "{query}". '
            'For example, "marriage status" should be similar to "divorced" or "ever divorsed" or "has a spouse", but not "marriage date". '
            'Reply with the index and name of that variable, without any explanation. For instance, "962. has a spouse" would be a valid response if that appears as a candidate. '
            'If none of the variable candidates seem appropriate, you should return "N/A" '
            'The variable candidates: '
            f'{top_matches}'
        )

        response = self.llm.chat.completions.create(
            model="gpt-4o-2024-08-06",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=40
        )

        result = response.choices[0].message.content.strip()

        if "N/A" in result:
            return "N/A"

        match = re.match(r'^(\d+)\.\s*(.+)$', result)
        if match:
            idx = int(match.group(1))
            # Return both the lhs and rhs corresponding to the best matching label.
            return lhs_labels[idx], rhs_labels[idx]

        return "N/A"
    

    def bin_numeric_series(self, series, n_bins=5):
        """
        Given a numeric Pandas Series, bins its values into exactly n_bins,
        using boundaries that come from the actual minimum and maximum values.
        The bins are labeled as "lower - upper" where the first bin is [min, b1)
        and the last bin is [b_{n-1}, max] (inclusive of the maximum).
        
        Returns:
            A Categorical Series with bin labels.
        """
        # Convert to integers (assuming the numeric domain is integral)
        min_val = int(series.min())
        max_val = int(series.max())
        # Compute boundaries: we want exactly n_bins intervals; using np.linspace over [min, max+1)
        boundaries = np.linspace(min_val, max_val + 1, n_bins + 1)
        # Round boundaries to integers (they should be nearly integers already)
        boundaries = np.round(boundaries).astype(int)
        # Ensure boundaries are unique; if rounding collapses bins, fall back to a manual integer range.
        if len(np.unique(boundaries)) - 1 < n_bins:
            boundaries = np.arange(min_val, max_val + 2)
            indices = np.linspace(0, len(boundaries) - 1, n_bins + 1, dtype=int)
            boundaries = boundaries[indices]
        
        # Create bin labels. For bins except the last, label as "lower - (upper-1)"; last bin: "lower - upper".
        labels = []
        for i in range(len(boundaries) - 1):
            lower = boundaries[i]
            if i < len(boundaries) - 2:
                upper = boundaries[i+1] - 1
            else:
                upper = boundaries[i+1]
            labels.append(f"{lower} - {upper}")
        
        # Use pd.cut to bin the numeric data. We use right=False so that bins are left-closed and right-open,
        # except the last bin which is closed on both ends.
        binned_series = pd.cut(series, bins=boundaries, labels=labels, include_lowest=True, right=False)
        return binned_series


    def get_variable_distribution(self, variable: str):
        """
        Given an existing DataFrame and a variable name, returns a tuple:
        (list of (category, percentage) pairs, sorted list of non-numeric responses discarded)
        Excludes responses in INVALID_RESPONSES.
        
        If over 50% of the responses (after excluding invalid ones) are numeric,
        the numeric portion is binned into exactly 7 bins (using domain boundaries),
        and any non-numeric responses are discarded and reported.
        
        Otherwise, computes the distribution over all valid responses.
        """
        # Create a mapping of lowercase column names to the actual column names.
        columns_lower = {col.lower(): col for col in self.gss_df.columns}
        var_lower = variable.lower()
        if var_lower not in columns_lower:
            raise ValueError(f"Variable '{variable}' not found in the DataFrame.")
        actual_variable = columns_lower[var_lower]

        # Exclude invalid responses in a case-insensitive manner.
        invalid_lower = {s.lower() for s in self.INVALID_RESPONSES}
        valid_series = self.gss_df[actual_variable].apply(lambda x: x.lower() if isinstance(x, str) else x)
        valid_series = valid_series[~valid_series.isin(invalid_lower)]
        
        # Convert values to numeric (non-numeric become NaN) while preserving the original index.
        numeric_initial = pd.to_numeric(valid_series, errors='coerce').reindex(valid_series.index)
        # Build a boolean mask for values in the range [0, 5000].
        mask = numeric_initial.notnull() & numeric_initial.between(0, 5000)
        numeric_values = numeric_initial[mask]
        
        n_total = len(valid_series)
        n_numeric = mask.sum()
        frac_numeric = n_numeric / n_total if n_total > 0 else 0
        
        if frac_numeric >= 0.5:
            # Over 50% are numeric: bin these numeric values.
            numeric_series = numeric_values.dropna()
            binned = self.bin_numeric_series(numeric_series, n_bins=7)
            #print("debug4: binned numeric domains.")
            distribution = binned.value_counts(normalize=True, dropna=False) * 100
            # Use the aligned mask to select non-numeric responses.
            non_numeric = sorted(set(valid_series[~mask]))
            return list(distribution.items()), non_numeric
        else:
            # Treat as categorical.
            distribution = valid_series.value_counts(normalize=True, dropna=False) * 100
            return list(distribution.items()), []


    def get_variable_distribution_with_constraints(self, variable: str, constraints: dict):
        """
        Given a DataFrame, a target variable, and a dictionary of constraints,
        this function validates and transforms each constraint value (using case-insensitive
        matching) and filters the DataFrame accordingly. It then computes the distribution
        for the target variable while preserving its full domain.
        
        For the target variable:
        - If over 50% of the valid responses are numeric, the numeric responses are binned into
            exactly 7 bins using boundaries from the full numeric domain; any bin that is missing in the
            filtered data will be returned with 0%.
        - Otherwise, the categorical distribution is computed over the full set of valid responses,
            ensuring that any category missing after filtering is still represented with 0%.
        
        It excludes responses in INVALID_RESPONSES.
        
        Returns:
            A tuple of (list of (category, percentage) pairs, sorted list of non-numeric responses discarded).
        """
        # Map lowercase column names to actual DataFrame column names.
        columns_lower = {col.lower(): col for col in self.gss_df.columns}
        var_lower = variable.lower()
        if var_lower not in columns_lower:
            raise ValueError(f"Variable '{variable}' not found in the DataFrame.")
        actual_variable = columns_lower[var_lower]
        print(f"in get gold distribution, locates variable {actual_variable}")

        # --- Validate and transform constraints ---
        new_constraints = {}
        for c_var, c_val in constraints.items():
            c_var_lower = c_var.lower()
            if c_var_lower not in columns_lower:
                raise ValueError(f"Constraint variable '{c_var}' not found in the DataFrame.")
            actual_c_var = columns_lower[c_var_lower]
            # Build the domain from the DataFrame and convert all values to lowercase strings.
            domain = {str(x).lower() for x in self.gss_df[actual_c_var].dropna().unique()}

            def validate_value(val):
                val_str = str(val).lower()
                # If the value is in the domain, return it.
                if val_str in domain:
                    return val_str
                # If it matches a "lower - upper" format, try to collect matching numeric domain values.
                m = re.match(r'^(\d+)\s*-\s*(\d+)$', str(val))
                if m:
                    lower_bound = int(m.group(1))
                    upper_bound = int(m.group(2))
                    bin_vals = [str(x).lower() for x in domain 
                                if str(x).isdigit() and lower_bound <= int(x) <= upper_bound]
                    if bin_vals:
                        return bin_vals  # return a list of matching values
                raise ValueError(f"Value '{val}' for constraint variable '{c_var}' is not in its domain: {domain}")

            if isinstance(c_val, list):
                transformed = []
                for x in c_val:
                    res = validate_value(x)
                    if isinstance(res, list):
                        transformed.extend(res)
                    else:
                        transformed.append(res)
                new_constraints[actual_c_var] = list(set(transformed))
            else:
                new_constraints[actual_c_var] = validate_value(c_val)
            print(f"processed constraints: {new_constraints}")

        # --- Filter the DataFrame based on new_constraints ---
        filtered_df = self.gss_df.copy()
        for c_var, c_val in new_constraints.items():
            if isinstance(c_val, list):
                filtered_df = filtered_df[
                    filtered_df[c_var].apply(lambda x: str(x).lower() if isinstance(x, str) else x)
                    .isin(c_val)
                ]
            else:
                filtered_df = filtered_df[
                    filtered_df[c_var].apply(lambda x: str(x).lower() if isinstance(x, str) else x) == c_val
                ]

        if actual_variable not in filtered_df.columns:
            raise ValueError(f"Variable '{variable}' not found in the DataFrame.")

        # --- Compute the full target domain from the entire DataFrame ---
        invalid_lower = {str(s).lower() for s in self.INVALID_RESPONSES}
        full_target_series = self.gss_df[actual_variable].apply(
            lambda x: str(x).lower() if isinstance(x, str) else x
        )
        full_target_series = full_target_series[~full_target_series.isin(invalid_lower)]

        # Determine if the target variable is mostly numeric.
        full_numeric = pd.to_numeric(full_target_series, errors='coerce')
        if full_numeric.notnull().sum() / len(full_target_series) >= 0.5:
            # For numeric target: compute full bins from the complete numeric data.
            full_numeric_clean = full_numeric.dropna()
            binned_full = self.bin_numeric_series(full_numeric_clean, n_bins=7)
            full_bins = list(binned_full.cat.categories)  # full set of bin labels
        else:
            full_target_domain = sorted(list(full_target_series.unique()))

        # --- Process the filtered target variable ---
        valid_series = filtered_df[actual_variable].apply(
            lambda x: str(x).lower() if isinstance(x, str) else x
        )
        valid_series = valid_series[~valid_series.isin(invalid_lower)]

        numeric_initial = pd.to_numeric(valid_series, errors='coerce')
        mask = numeric_initial.notnull() & numeric_initial.between(0, 5000)
        n_total = len(valid_series)
        n_numeric = mask.sum()
        frac_numeric = n_numeric / n_total if n_total > 0 else 0

        if frac_numeric >= 0.5:
            # Use numeric binning.
            numeric_series = numeric_initial[mask].dropna()
            binned = self.bin_numeric_series(numeric_series, n_bins=7)
            counts = binned.value_counts(dropna=False)
            # Reindex with the full bins to include bins with 0 counts.
            counts = counts.reindex(full_bins, fill_value=0)
            total_count = counts.sum()
            distribution = [
                (bin_label, (count / total_count) * 100 if total_count > 0 else 0)
                for bin_label, count in counts.items()
            ]
            non_numeric = sorted(set(valid_series[~mask]))
            return distribution, non_numeric
        else:
            # Use categorical distribution.
            counts = valid_series.value_counts(dropna=False)
            # Reindex with the full target domain to include any missing categories.
            counts = counts.reindex(full_target_domain, fill_value=0)
            total_count = counts.sum()
            distribution = [
                (cat, (count / total_count) * 100 if total_count > 0 else 0)
                for cat, count in counts.items()
            ]
            return distribution, []



    def get_constraints(self, node):
        """
        Convert the node's bound variables into constraints.
        For each bound variable, check if its value(s) appear in the variable's domain (from self.gss_df)
        if available. If the variable is not present in self.gss_df (or its domain is empty), a warning is logged
        and the provided value(s) are accepted as-is.
        
        For each value:
        - If the value is in the domain, accept it.
        - If not, and if it matches a bin format (e.g. "28 - 37"), then include all numeric domain values
            that fall in that range.
        - If neither condition holds, log a warning and ignore that constraint.
        Returns a dictionary mapping variable names to accepted constraint value(s).
        """
        import re
        constraints = {}
        for var in node.bound:
            if not var.value:
                continue
            
            name = self.lookup_varname_from_description(var.name, "/Users/ruoxiliu/Desktop/Argo_research/reasoning/sense/backtracking/GSS2022-varbook.txt")
            # Try to get the domain; if not found, log warning and accept the provided value(s)
            if name in self.gss_df.columns:
                domain = set(self.gss_df[name].dropna().unique())
                print(f"In get constraints, successfully locates the variable {name}")
                print(f"obtained domain: {domain}")
            else:
                with open(self.log_pth, "a") as log_file:
                    print(f"[err] Constraint variable '{var.name}' not found in DataFrame. Using provided value(s) without validation.\n")
                domain = set()

            accepted = []
            
            def validate_value(val):
                # If domain exists and val is in it, return val.
                if domain and (val in domain):
                    return val
                # If domain is empty, simply return the provided value.
                if not domain:
                    return val
                # Otherwise, check if val matches a bin format, e.g. "28 - 37"
                m = re.match(r'^(\d+)\s*-\s*(\d+)$', str(val))
                if m:
                    lower = int(m.group(1))
                    upper = int(m.group(2))
                    bin_vals = [x for x in domain if isinstance(x, (int, float)) and lower <= x <= upper]
                    if bin_vals:
                        return bin_vals  # return a list
                raise ValueError(f"Value '{val}' for constraint variable '{var.name}' is not in its domain: {domain}")
            
            try:
                if isinstance(var.value, list):
                    transformed = []
                    for x in var.value:
                        res = validate_value(x)
                        if isinstance(res, list):
                            transformed.extend(res)
                        else:
                            transformed.append(res)
                    accepted = list(set(transformed))
                else:
                    accepted = validate_value(var.value)
            except ValueError as e:
                # Log the error and skip this constraint.
                with open(self.log_pth, "a") as log_file:
                    log_file.write(f"[warn] {e}. Ignoring constraint for variable '{var.name}'.\n")
                continue
            
            constraints[var.name] = accepted if isinstance(accepted, list) and len(accepted) > 1 else accepted

        print(f"constraints obtained for node {node}: ", constraints)
        return constraints


    
    def lookup_varname_from_description(self, description: str, varbook_path: str) -> str:
        description_lower = description.lower()
        with open(varbook_path, 'r') as f:
            for line in f:
                line = line.strip()
                if ':' in line:
                    lhs, rhs = line.split(':', 1)
                    if rhs.strip().lower() == description_lower:
                        return lhs.strip()
                else:
                    # If no colon, assume the line is both lhs and rhs.
                    if line.strip().lower() == description_lower:
                        return line.strip()
        return None

    
    def compute_cross_entropy_loss(self, target_desc: str, predicted_distribution: list, constraints_desc: dict, varbook_path: str) -> float:
        """
        Given a target variable description, a predicted probability distribution (list of floats),
        and constraints (as a dict mapping constraint variable descriptions to desired value descriptions),
        this method uses the varbook to map descriptions to variable names, retrieves the gold distribution 
        using get_variable_distribution_with_constraints, and computes the cross entropy loss between the 
        gold distribution (converted from percentages to probabilities) and the predicted distribution.
        
        Assumptions:
        - The gold distribution's category order corresponds to the order of predicted_distribution.
        - Predicted_distribution values are positive and sum to 1.
        
        Returns:
        Cross entropy loss as a float.
        """

        # Lookup target variable name from description.
        target_var_name = self.lookup_varname_from_description(target_desc, varbook_path)
        if target_var_name is None:
            raise ValueError(f"Could not find a matching variable for target description '{target_desc}' in the varbook.")

        # Build constraints dictionary with variable names as keys.
        constraints_names = {}
        for cond_desc, cond_value_desc in constraints_desc.items():
            cond_var_name = self.lookup_varname_from_description(cond_desc, varbook_path)
            if cond_var_name is None:
                raise ValueError(f"Could not find a matching variable for constraint description '{cond_desc}' in the varbook.")
            # For the value, we assume that the value description itself is used as the value (or you can also map it)
            cond_value = cond_value_desc
            # If no mapping for the value is found, fall back to using the provided description.
            if cond_value is None:
                cond_value = cond_value_desc
            constraints_names[cond_var_name] = cond_value

        # Retrieve the gold distribution (and ignored non-numeric responses) using your existing method.
        gold_distribution, non_numeric = self.get_variable_distribution_with_constraints(target_var_name, constraints_names)
        print("gold distribution: ", gold_distribution)
        print("non-numeric: ", non_numeric)
        if not gold_distribution:
            print("constraints: ", constraints_names)
            print("target_var_name: ", target_var_name)
            raise ValueError("Gold distribution retrieval failed or is empty.")

        # Convert the gold distribution percentages into probabilities.
        gold_probs = [pct / 100.0 for category, pct in gold_distribution]

        if len(predicted_distribution) != len(gold_probs):
            print("predicted_distribution: ", predicted_distribution)
            print("gold_distribution: ", gold_distribution)
            raise ValueError("Mismatch in length: predicted distribution and gold distribution have different numbers of categories.")

        # Compute cross entropy loss: - sum_i (gold_prob[i] * log(predicted_prob[i]))
        eps = 1e-12  # small epsilon for numerical stability
        predicted_distribution = [max(p, eps) for p in predicted_distribution]
        cross_entropy = -np.sum([g * np.log(p) for g, p in zip(gold_probs, predicted_distribution)])
        return cross_entropy

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
            
    
    # Recursively backtrack to get the distribution of the target variable at the root
    # Calculation is simple -- basically take an average of the children's target distributions, use the prob of the childrens as weight
    # Eg. if node has child 1 with prob 0.5, child 2 with prob 0.5, child 1 has target var distribution as ["rainy", "sunny"] with [0.2, 0.8], and child 1 has target var distribution as ["rainy", "sunny"] with [0.3, 0.7]
    # Then node should have target var distribution as ["rainy", "sunny"] with [0.25, 0.75]
    def recursive_backtrack(self, node):
        #recursion endpoint
        if not node.children:

            print(f"prob is {node.target.prob} for node {node.name}")


            try:
                constraints_desc = self.get_constraints(node)
                # For our evaluation, we assume the target description is node.target.name.
                # (In your setup, this should correspond to the varbook description.)
                ce_loss = self.compute_cross_entropy_loss(
                    target_desc=node.target.name,
                    predicted_distribution=node.target.prob,
                    constraints_desc=constraints_desc,
                    varbook_path="/Users/ruoxiliu/Desktop/Argo_research/reasoning/sense/backtracking/GSS2022-varbook.txt"
                )
                node_depth = len(node.name)
                # Write the (CE loss, depth) pair to file.
                ce_output_path = "/Users/ruoxiliu/Desktop/Argo_research/reasoning/sense/backtracking/target_CE_depth.txt"
                with open(ce_output_path, "a") as ce_file:
                    ce_file.write(f"{node_depth}\t{ce_loss}\n")
                node.CE = ce_loss
            except Exception as e:
                print(f"got error when deriving cross entropy: {e}")



            return
        
        accumulated_prob = []
        try:
            #Otherwise, for non-leaf nodes, need to directly estimate target for CE
            updated_node, updated_target = self.craft_target_distribution_S(node)
            node_depth = len(node.name) 
            constraints_desc = self.get_constraints(updated_node)
            ce_loss = self.compute_cross_entropy_loss(
                target_desc=updated_node.target.name,
                predicted_distribution=updated_node.target.prob,
                constraints_desc=constraints_desc,
                varbook_path="/Users/ruoxiliu/Desktop/Argo_research/reasoning/sense/backtracking/GSS2022-varbook.txt"
            )
            ce_output_path = "/Users/ruoxiliu/Desktop/Argo_research/reasoning/sense/backtracking/target_CE_depth.txt"
            with open(ce_output_path, "a") as ce_file:
                ce_file.write(f"{node_depth}\t{ce_loss}\n")

            node.CE = ce_loss
        except Exception as e:
                print(f"got error when deriving cross entropy: {e}")


        #find the target variable (so that we know the probability of reaching each child from the parent node)
        target_var = None
        for var_parent in node.unbound:
            for var_child in node.children[0].bound:
                if var_parent.name == var_child.name:
                    target_var = var_parent
                    
        for idx, child in enumerate(node.children):
            
            #recursively establish the prob of the children
            self.recursive_backtrack(child)

            if not accumulated_prob:
                accumulated_prob = [p * target_var.prob[idx] for p in child.target.prob]
            else:
                for i in range(len(accumulated_prob)):
                    accumulated_prob[i] += child.target.prob[i] * target_var.prob[idx]
 
        node.target.prob = accumulated_prob
        print(f"prob is {node.target.prob} for node {node.name}")


         
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
    
    def spawn_children(self, node):
        unbound_var = node.ready_to_spawn()
        if unbound_var is None:
            return []
        
        remaining_unbound_vars = [q for q in node.unbound if q != unbound_var]

        for idx, value in enumerate(unbound_var.value):

            # define name. eg: "341" -> "3411"
            child_name = node.name + str(idx)

            child_prob = node.prob * unbound_var.prob[idx]

            # Create a new fact
            new_bound_var = Variable(
                name=unbound_var.name,
                type=unbound_var.type,
                value=[value],
            )
            child_bound_vars = list(node.bound)
            child_bound_vars.append(new_bound_var)

            child_node = S_node(
                name=child_name,
                target=node.target,
                unbound=remaining_unbound_vars,
                bound=child_bound_vars,
                prob=child_prob
            )
            node.children.append(child_node)
            child_node.parent = node
        
        return node.children
        
 
agent = AgentS1("/Users/ruoxiliu/Desktop/Argo_research/reasoning/sense/backtracking/1.json")
agent.s_agent_entry(50, False)
print("The final calculated target:", agent.s_root.target)