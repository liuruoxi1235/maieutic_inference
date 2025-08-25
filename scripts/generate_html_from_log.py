import os
import re
import json
import ast
from jinja2 import Environment, Template

# -----------------------------------------------------------
# Helper: Case-insensitive key lookup in dictionaries.
# -----------------------------------------------------------
def get_ci_value(d, key):
    """
    Given a dictionary d, return the value for the first key matching
    key case-insensitively, or None if not found.
    """
    if isinstance(d, dict):
        for k, v in d.items():
            if k.lower() == key.lower():
                return v
    return None

# -----------------------------------------------------------
# Helper: Extract balanced JSON segments (supports nested braces)
# -----------------------------------------------------------
def extract_json_segments(text):
    """
    Scans text and returns a list of segments as tuples (is_json, segment_text),
    where is_json is True if the segment is enclosed in balanced curly braces,
    and False for plain text.
    Uses a stack-based approach to support nested braces.
    """
    segments = []
    stack = []
    last_index = 0
    start = None
    for i, char in enumerate(text):
        if char == '{':
            if not stack:
                if i > last_index:
                    segments.append((False, text[last_index:i]))
                start = i
            stack.append(i)
        elif char == '}':
            if stack:
                stack.pop()
                if not stack and start is not None:
                    segments.append((True, text[start:i+1]))
                    last_index = i + 1
                    start = None
    if last_index < len(text):
        segments.append((False, text[last_index:]))
    return segments

# -----------------------------------------------------------
# Custom filter: Parse a JSON segment.
# -----------------------------------------------------------
def parse_json_segment(text):
    text = text.strip()
    parsed = None
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        try:
            parsed = ast.literal_eval(text)
        except Exception:
            parsed = None
    return parsed

# -----------------------------------------------------------
# STEP 1: Parse the Log File (ignoring lines not in expected form)
# -----------------------------------------------------------
def parse_log_file(filename):
    """
    Reads the log file and returns (page_title, entries).

    Expected (case-insensitive) lines:
      [title] : Some Title Here
      [main x] [step x] : <message>
      [main x] [GSS replacement internal steps] : <message>
      [main x] [content] : <json object>
      [main x] [prompt] : <json array of role-message pairs>

    Lines not starting with "[title]" or "[main" are ignored.
    For content and prompt lines, the JSON is parsed via json.loads with a fallback.
    """
    page_title = "Rollout Log Viewer"
    entries = []
    order = 0

    if not os.path.exists(filename):
        print("ERROR: Log file not found at", filename)
        return page_title, entries

    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            # Only process lines starting with [title] or [main]
            if not (line.lower().startswith("[title]") or line.lower().startswith("[main")):
                continue
            if not line:
                continue

            # [title] line
            match_title = re.match(r".*\[title\]\s*:\s*(.*)", line, flags=re.IGNORECASE)
            if match_title:
                page_title = match_title.group(1).strip()
                continue

            # [step] line
            match_step = re.match(r"\[main\s+([0-9]+)\]\s+\[step\s+([0-9]+)\]\s*:\s*(.*)", line, flags=re.IGNORECASE)
            if match_step:
                main_code = match_step.group(1)
                step_num = match_step.group(2)
                message = match_step.group(3).strip()
                entries.append({
                    "main": main_code,
                    "type": "step",
                    "step": step_num,
                    "message": message,
                    "order": order
                })
                order += 1
                continue

            # [GSS replacement internal steps] line (treated as step)
            match_internal = re.match(r"\[main\s+([0-9]+)\]\s+\[GSS replacement internal steps\]\s*:\s*(.*)", 
                                       line, flags=re.IGNORECASE)
            if match_internal:
                main_code = match_internal.group(1)
                message = match_internal.group(2).strip()
                entries.append({
                    "main": main_code,
                    "type": "step",
                    "step": "internal",
                    "message": message,
                    "order": order
                })
                order += 1
                continue

            # [content] line
            match_content = re.match(r"\[main\s+([0-9]+)\]\s+\[content\]\s*:\s*(.*)", line, flags=re.IGNORECASE)
            if match_content:
                main_code = match_content.group(1)
                content_str = match_content.group(2).strip()
                parsed_obj = None
                try:
                    parsed_obj = json.loads(content_str)
                except json.JSONDecodeError:
                    try:
                        parsed_obj = ast.literal_eval(content_str)
                    except Exception:
                        parsed_obj = None
                entries.append({
                    "main": main_code,
                    "type": "content",
                    "message": content_str,
                    "parsed_content": parsed_obj,
                    "order": order
                })
                order += 1
                continue

            # [prompt] line
            match_prompt = re.match(r"\[main\s+([0-9]+)\]\s+\[prompt\]\s*:\s*(.*)", line, flags=re.IGNORECASE)
            if match_prompt:
                main_code = match_prompt.group(1)
                prompt_str = match_prompt.group(2).strip()
                parsed_obj = None
                try:
                    parsed_obj = json.loads(prompt_str)
                except json.JSONDecodeError:
                    try:
                        parsed_obj = ast.literal_eval(prompt_str)
                    except Exception:
                        parsed_obj = None
                entries.append({
                    "main": main_code,
                    "type": "prompt",
                    "message": prompt_str,
                    "parsed_content": parsed_obj,
                    "order": order
                })
                order += 1
                continue

    return page_title, entries

# -----------------------------------------------------------
# STEP 2: Build the Recursive Tree Structure
# -----------------------------------------------------------
def build_tree(entries):
    """
    Builds a tree of nodes from the log entries.
    Each node is a dictionary with keys:
       'main': node code (string)
       'steps': list of step entries
       'contents': list of content entries
       'prompts': list of prompt entries
       'all': merged list of all entries (steps + contents + prompts) sorted by order
       'children': list of child nodes

    A node B is a child of node A if B's main code equals A's main code with extra digits appended.
    """
    nodes = {}
    for e in entries:
        code = e["main"]
        if code not in nodes:
            nodes[code] = {"main": code, "steps": [], "contents": [], "prompts": [], "all": [], "children": []}
        if e["type"] == "step":
            nodes[code]["steps"].append(e)
        elif e["type"] == "content":
            nodes[code]["contents"].append(e)
        elif e["type"] == "prompt":
            nodes[code]["prompts"].append(e)

    for code, node in nodes.items():
        combined_all = node["steps"] + node["contents"] + node["prompts"]
        combined_all.sort(key=lambda x: x["order"])
        node["all"] = combined_all

    for code, node in nodes.items():
        if code == "0":
            continue
        parent_code = code[:-1]
        if parent_code in nodes:
            nodes[parent_code]["children"].append(node)
        else:
            print(f"Warning: No parent found for node '{code}'")
    return nodes.get("0", {})

# -----------------------------------------------------------
# STEP 3: Jinja2 Template with Merged Ordering and Checkbox Filtering
# -----------------------------------------------------------
# In this version, each node is rendered as a card with a single merged section.
# The merged section ("all") displays entries in the order they appeared in the log file.
# Each entry gets a CSS class ("type-step", "type-content", or "type-prompt").
# Three independent checkboxes ("Steps" and "Contents" are checked by default; "Prompts" is unchecked by default)
# filter the visible entries.
# We also revise the rendering of variables so that if a variable entry has a "Prob" list,
# each value is displayed along with its corresponding probability.
template_str = r"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>{{ page_title }}</title>
    <style>
        /* Overall styling */
        body {
            font-family: Arial, sans-serif;
            background-color: #f5f5f5;
            margin: 20px;
            line-height: 1.6;
        }
        .container {
            max-width: 1200px;
            margin: auto;
            padding: 30px;
            background-color: #fff;
            border-radius: 8px;
            box-shadow: 0 0 15px rgba(0,0,0,0.1);
        }
        h1 {
            text-align: center;
            color: #333;
            margin-bottom: 25px;
        }
        .log-card {
            background-color: #f9f9f9;
            border: 1px solid #ddd;
            border-radius: 5px;
            padding: 15px;
            margin: 15px 0;
            box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        }
        .log-card h2 {
            margin-top: 0;
            color: #4285f4;
            font-size: 20px;
        }
        .log-entry {
            margin-left: 10px;
            margin: 5px 0;
            padding: 5px 10px;
            font-size: 14px;
        }
        .type-step {
            background-color: #fff;
            border-left: 4px solid #34a853;
            border-radius: 3px;
        }
        .type-content {
            background-color: #fff;
            border-left: 4px solid #f44336;
            border-radius: 3px;
        }
        .type-prompt {
            background-color: #eef;
            border-left: 4px solid #a66;
            border-radius: 3px;
        }
        .node-controls {
            margin-bottom: 10px;
        }
        .node-controls label {
            margin-right: 15px;
            font-size: 14px;
        }
        .section-content {
            margin-top: 10px;
        }
        .child-container {
            margin-left: 30px;
            margin-top: 10px;
            padding-left: 15px;
            border-left: 2px dashed #ddd;
        }
        .toggle-btn {
            display: inline-block;
            padding: 6px 12px;
            margin: 10px 0;
            background-color: #4285f4;
            color: #fff;
            border: none;
            border-radius: 4px;
            font-size: 14px;
            cursor: pointer;
            box-shadow: 0 2px 5px rgba(0,0,0,0.2);
            transition: background-color 0.3s;
        }
        .toggle-btn:hover {
            background-color: #357ae8;
        }
        /* Parsed content styling */
        .parsed-content-box {
            margin: 5px 0;
            padding: 10px;
            border: 1px solid #ccc;
            background-color: #fafafa;
            border-radius: 5px;
        }
        .parsed-content-row {
            display: flex;
            flex-wrap: wrap;
            margin: 5px 0;
        }
        .parsed-section {
            flex: 1;
            margin: 5px;
            padding: 8px;
            border-radius: 4px;
            min-width: 200px;
        }
        .prob-section { background-color: #fffae6; }
        .target-section { background-color: #e6f3ff; }
        .unbound-section { background-color: #fffbe6; }
        .bound-section { background-color: #e6ffe6; }
        .parsed-title {
            font-weight: bold;
            margin-bottom: 5px;
        }
        .parsed-item {
            margin-left: 15px;
        }
        .parsed-item-sub {
            margin-left: 25px;
        }
        .prompt-segment-text {
            white-space: pre-wrap;
        }
    </style>
    <script>
        // Filter merged entries within a node based on checkbox states.
        function filterEntries(prefix) {
            var showStep = document.getElementById('chk-steps-' + prefix).checked;
            var showContent = document.getElementById('chk-contents-' + prefix).checked;
            var showPrompt = document.getElementById('chk-prompts-' + prefix).checked;
            var container = document.getElementById('all-' + prefix);
            if (!container) return;
            var entries = container.getElementsByClassName('log-entry');
            for (var i = 0; i < entries.length; i++) {
                var entry = entries[i];
                if (entry.classList.contains('type-step')) {
                    entry.style.display = showStep ? 'block' : 'none';
                }
                if (entry.classList.contains('type-content')) {
                    entry.style.display = showContent ? 'block' : 'none';
                }
                if (entry.classList.contains('type-prompt')) {
                    entry.style.display = showPrompt ? 'block' : 'none';
                }
            }
        }
        // When a checkbox is toggled, call filterEntries.
        function toggleType(type, prefix) {
            filterEntries(prefix);
        }
        // Toggle child node display.
        function toggleChild(nodeId) {
            var element = document.getElementById(nodeId);
            if (!element) return;
            element.style.display = (element.style.display === "none") ? "block" : "none";
        }
    </script>
</head>
<body>
    <div class="container">
        <h1>{{ page_title }}</h1>
        {# Helper macro: display a value (string or list) line-by-line #}
        {% macro display_value(val) %}
            {% if val is sequence and (val is not string) %}
                {% for item in val %}
                    <div class="parsed-item-sub">{{ item }}</div>
                {% endfor %}
            {% else %}
                {{ val }}
            {% endif %}
        {% endmacro %}
        
        {# New macro: render a variable entry with optional probability distribution #}
        {% macro render_variable(var) %}
            <div class="parsed-item">
                <strong>{{ var.Name }}</strong>:
                {% if var.Value is defined %}
                    {% if var.Prob is defined and (var.Prob is sequence) and (var.Prob|length == var.Value|length) %}
                        {% for i in range(var.Value|length) %}
                            <div class="parsed-item-sub">{{ var.Value[i] }} : {{ var.Prob[i] }}</div>
                        {% endfor %}
                    {% else %}
                        {{ display_value(var.Value) }}
                    {% endif %}
                {% else %}
                    N/A
                {% endif %}
            </div>
        {% endmacro %}
        
        {# Macro to render parsed JSON with fixed labels, using render_variable for each variable #}
        {% macro render_parsed_content(parsed) %}
        <div class="parsed-content-box">
            <div class="parsed-content-row">
                <div class="parsed-section prob-section">
                    <div class="parsed-title">Probability</div>
                    <div class="parsed-item">
                        {% set prob_val = get_ci_value(parsed, "prob") %}
                        {% if prob_val is not none %}
                            {% if prob_val is string or prob_val is number %}
                                {{ prob_val }}
                            {% else %}
                                N/A
                            {% endif %}
                        {% else %}
                            N/A
                        {% endif %}
                    </div>
                </div>
                <div class="parsed-section target-section">
                    <div class="parsed-title">Target Variables</div>
                    {% set target = get_ci_value(parsed, "target") %}
                    {% if target is defined and target %}
                        {% if target|length > 0 %}
                            {% for t in target %}
                                {{ render_variable(t) }}
                            {% endfor %}
                        {% else %}
                            <div class="parsed-item">No target items.</div>
                        {% endif %}
                    {% else %}
                        <div class="parsed-item">No target field.</div>
                    {% endif %}
                </div>
                <div class="parsed-section unbound-section">
                    <div class="parsed-title">Unbound Conditional Variables</div>
                    {% set unbound = get_ci_value(parsed, "unbound_cond") %}
                    {% if unbound is defined and unbound %}
                        {% if unbound|length > 0 %}
                            {% for uc in unbound %}
                                {{ render_variable(uc) }}
                            {% endfor %}
                        {% else %}
                            <div class="parsed-item">No unbound conditions.</div>
                        {% endif %}
                    {% else %}
                        <div class="parsed-item">No unbound field.</div>
                    {% endif %}
                </div>
                <div class="parsed-section bound-section">
                    <div class="parsed-title">Bound Conditional Variables</div>
                    {% set bound = get_ci_value(parsed, "bound_cond") %}
                    {% if bound is defined and bound %}
                        {% if bound|length > 0 %}
                            {% for bc in bound %}
                                {{ render_variable(bc) }}
                            {% endfor %}
                        {% else %}
                            <div class="parsed-item">No bound conditions.</div>
                        {% endif %}
                    {% else %}
                        <div class="parsed-item">No bound field.</div>
                    {% endif %}
                </div>
            </div>
        </div>
        {% endmacro %}
        
        {# Macro to render embedded JSON within prompt text #}
        {% macro render_prompt_content(text) %}
            {% set segments = text | extract_json_segments %}
            {% for is_json, seg in segments %}
                {% if is_json %}
                    {% set parsed = seg | parse_json_segment %}
                    {% if parsed is not none %}
                        {{ render_parsed_content(parsed) }}
                    {% else %}
                        <div class="prompt-segment-text">{{ seg }}</div>
                    {% endif %}
                {% else %}
                    <div class="prompt-segment-text">{{ seg }}</div>
                {% endif %}
            {% endfor %}
        {% endmacro %}
        
        {# Macro to render a single prompt entry (an array of role-message pairs) #}
        {% macro render_prompt_entry(p) %}
            {% if p.parsed_content %}
                {% for pair in p.parsed_content %}
                    <div class="log-prompt type-prompt">
                        <strong>{{ pair.role | capitalize }}</strong><br/>
                        {{ render_prompt_content(pair.content) }}
                    </div>
                {% endfor %}
            {% else %}
                <div class="log-prompt type-prompt">
                    <strong>Prompt:</strong>
                    <div class="prompt-segment-text">{{ p.message }}</div>
                </div>
            {% endif %}
        {% endmacro %}
        
        {# Macro to render all prompt entries #}
        {% macro render_prompts(prompts) %}
            {% for p in prompts %}
                {{ render_prompt_entry(p) }}
            {% endfor %}
        {% endmacro %}
        
        {# Recursive macro to render a node #}
        {% macro render_node(node, prefix) %}
            <div class="log-card">
                <h2>Node {{ node.main }}</h2>
                
                <!-- Checkbox controls -->
                <div class="node-controls">
                    <label>
                        <input type="checkbox" id="chk-steps-{{ prefix }}" onchange="toggleType('steps', '{{ prefix }}')" checked> Steps
                    </label>
                    <label>
                        <input type="checkbox" id="chk-contents-{{ prefix }}" onchange="toggleType('contents', '{{ prefix }}')" checked> Contents
                    </label>
                    <label>
                        <input type="checkbox" id="chk-prompts-{{ prefix }}" onchange="toggleType('prompts', '{{ prefix }}')"> Prompts
                    </label>
                </div>
                
                <!-- Merged section: entries are shown in log-file order -->
                <div id="all-{{ prefix }}" class="section-content">
                    {% for entry in node.all %}
                        {% if entry.type == "step" %}
                            <div class="log-entry type-step">
                                <strong>[step {{ entry.step }}]</strong> : {{ entry.message }}
                            </div>
                        {% elif entry.type == "content" %}
                            <div class="log-entry type-content">
                                <div class="log-content">
                                    {% if entry.parsed_content %}
                                        {{ render_parsed_content(entry.parsed_content) }}
                                    {% else %}
                                        {{ entry.message }}
                                    {% endif %}
                                </div>
                            </div>
                        {% elif entry.type == "prompt" %}
                            <div class="log-entry type-prompt">
                                <div class="log-prompt">
                                    {% if entry.parsed_content %}
                                        {{ render_prompts([entry]) }}
                                    {% else %}
                                        {{ entry.message }}
                                    {% endif %}
                                </div>
                            </div>
                        {% endif %}
                    {% endfor %}
                </div>
                
                <!-- Apply filtering -->
                <script>
                    filterEntries("{{ prefix }}");
                    function filterEntries(prefix) {
                        var showStep = document.getElementById('chk-steps-' + prefix).checked;
                        var showContent = document.getElementById('chk-contents-' + prefix).checked;
                        var showPrompt = document.getElementById('chk-prompts-' + prefix).checked;
                        var container = document.getElementById('all-' + prefix);
                        if (!container) return;
                        var entries = container.getElementsByClassName('log-entry');
                        for (var i = 0; i < entries.length; i++) {
                            var entry = entries[i];
                            if (entry.classList.contains('type-step')) {
                                entry.style.display = showStep ? 'block' : 'none';
                            }
                            if (entry.classList.contains('type-content')) {
                                entry.style.display = showContent ? 'block' : 'none';
                            }
                            if (entry.classList.contains('type-prompt')) {
                                entry.style.display = showPrompt ? 'block' : 'none';
                            }
                        }
                    }
                </script>
                
                <!-- Child nodes -->
                {% if node.children|length > 0 %}
                    <div class="child-container">
                        {% for child in node.children %}
                            <div class="child-node-wrapper">
                                <button class="toggle-btn" onclick="toggleChild('node-{{ child.main }}')">
                                    Show/Hide Node {{ child.main }}
                                </button>
                                <div id="node-{{ child.main }}" style="display: none;">
                                    {{ render_node(child, prefix ~ '-' ~ child.main) }}
                                </div>
                            </div>
                        {% endfor %}
                    </div>
                {% endif %}
            </div>
        {% endmacro %}
        
        {% if tree %}
            {{ render_node(tree, tree.main) }}
        {% else %}
            <p>No log entries found.</p>
        {% endif %}
    </div>
</body>
</html>
"""

# -----------------------------------------------------------
# Register custom filters in the Jinja2 Environment and Generate HTML
# -----------------------------------------------------------
def generate_html(log_file, output_html_file):
    page_title, raw_entries = parse_log_file(log_file)
    if not raw_entries:
        print(f"No valid log entries found in {log_file}")
        return

    tree = build_tree(raw_entries)
    # Compute merged "all" for each node
    def add_all_field(node):
        node["all"] = sorted(node["steps"] + node["contents"] + node["prompts"], key=lambda x: x["order"])
        for child in node.get("children", []):
            add_all_field(child)
    add_all_field(tree)

    env = Environment()
    env.filters["extract_json_segments"] = extract_json_segments
    env.filters["parse_json_segment"] = parse_json_segment
    template = env.from_string(template_str)
    rendered_html = template.render(page_title=page_title, tree=tree, get_ci_value=get_ci_value)
    
    with open(output_html_file, "w", encoding="utf-8") as f:
        f.write(rendered_html)
    print(f"Rollout HTML file generated: {output_html_file}")

# -----------------------------------------------------------
# MAIN EXECUTION
# -----------------------------------------------------------
if __name__ == "__main__":
    log_file_path = "/Users/ruoxiliu/Desktop/Argo_research/reasoning/sense/backtracking/test_trail_free_1.txt"
    output_html_file = "test_trail_free_1.html"
    generate_html(log_file_path, output_html_file)
