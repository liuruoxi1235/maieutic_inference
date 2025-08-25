#!/usr/bin/env python3
import os
import re
import json
import ast
import argparse
from jinja2 import Template

def parse_sample_trees(log_file_path):
    """
    Scan the log file for lines like:
      [main sampling 1] Full tree JSON:
      {
        ...nested JSON...
      }
    and return a dict { sample_index: tree_dict }.
    Uses brace‐counting so it stops exactly at the matching closing brace.
    """
    sample_trees = {}
    marker_re = re.compile(r'^\[main sampling\s+(\d+)\]\s+Full tree JSON:')
    with open(log_file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    i = 0
    n = len(lines)
    while i < n:
        m = marker_re.match(lines[i])
        if m:
            sample_idx = int(m.group(1))
            i += 1
            # Skip until the first '{'
            while i < n and '{' not in lines[i]:
                i += 1
            if i >= n:
                break
            # Collect lines until braces are balanced
            brace_count = 0
            json_lines = []
            while i < n:
                line = lines[i]
                brace_count += line.count('{')
                brace_count -= line.count('}')
                json_lines.append(line)
                i += 1
                if brace_count == 0:
                    break
            json_str = "".join(json_lines).strip()
            try:
                tree_obj = json.loads(json_str)
            except json.JSONDecodeError:
                tree_obj = ast.literal_eval(json_str)
            sample_trees[sample_idx] = tree_obj
        else:
            i += 1
    return sample_trees

def parse_final_prob(log_file_path):
    """
    Scan for the last “[final prob] …” line and return its parsed JSON list.
    """
    final = None
    prog = re.compile(r'^\[final prob\]\s*(.*)')
    with open(log_file_path, "r", encoding="utf-8") as f:
        for line in f:
            m = prog.match(line)
            if m:
                payload = m.group(1).strip()
                try:
                    final = json.loads(payload)
                except json.JSONDecodeError:
                    final = ast.literal_eval(payload)
    return final

HTML_TEMPLATE = r"""
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Sampling Tree Viewer (D3)</title>
  <style>
    body { font-family: sans-serif; margin:0; display:flex; flex-direction:column; height:100vh; }
    #controls { padding:10px; background:#fafafa; border-bottom:1px solid #ddd; }
    #main { display:flex; flex:1; overflow:hidden; }
    #tree-area { flex:2; position:relative; }
    #details { flex:1; border-left:1px solid #ddd; padding:10px; background:#f5f5f5; overflow:auto; }
    svg { width:100%; height:100%; }
    .link { fill:none; stroke-width:1.5px; }
    .link.default { stroke:#555; }
    .link.active  { stroke:red; }
    .node circle { fill:#fff; stroke:steelblue; stroke-width:1.5px; cursor:pointer; }
    .node text { font-size:12px; pointer-events:none; }
    #sample-num { font-weight:bold; margin-left:8px; }
    #final-prob-section { padding:10px; background:#fff; border-top:1px solid #ddd; }
    pre { white-space:pre-wrap; word-wrap:break-word; }
  </style>
</head>
<body>

  <div id="controls">
    <label>Sample #:
      <input type="range" id="sample-slider" min="1" max="{{ max_idx }}" value="1">
    </label>
    <span id="sample-num">1 / {{ max_idx }}</span>
  </div>

  <div id="main">
    <div id="tree-area"><svg></svg></div>
    <div id="details">
      <h2>Node Details</h2>
      <div id="node-json">Click a circle to see details here.</div>
    </div>
  </div>

  <div id="final-prob-section">
    <h3>Final Root Distribution</h3>
    <pre id="final-prob-display">(no data)</pre>
  </div>

  <script src="https://d3js.org/d3.v7.min.js"></script>
  <script>
    // Embed the sample→tree mapping and final distribution
    const sampleTrees = {{ sample_trees_json | safe }};
    const finalProb   = {{ final_prob_json   | safe }};
    const maxSample   = {{ max_idx }};
    const slider      = document.getElementById('sample-slider');
    const sampleNum   = document.getElementById('sample-num');
    slider.max        = maxSample;
    sampleNum.textContent = `1 / ${maxSample}`;

    // Show the final distribution
    if (Array.isArray(finalProb)) {
      document.getElementById("final-prob-display").textContent =
        JSON.stringify(finalProb, null, 2);
    } else {
      document.getElementById("final-prob-display").textContent =
        "(none)";
    }

    const svg        = d3.select("svg");
    const width      = svg.node().clientWidth;
    const height     = svg.node().clientHeight;
    const zoomGroup = svg.append("g").attr("class","zoom-group");
    const gLinks    = zoomGroup.append("g");
    const gNodes    = zoomGroup.append("g");
    const treeLayout = d3.tree().size([height - 20, width - 160]);

    svg.call(
        d3.zoom()
          .scaleExtent([0.5, 3])           // min/max zoom
          .on("zoom", (event) => {
            zoomGroup.attr("transform", event.transform);
          })
      );
    slider.addEventListener("input", () => drawTree(+slider.value));
    drawTree(1);

    function drawTree(idx) {
      sampleNum.textContent = `${idx} / ${maxSample}`;
      gLinks.selectAll("*").remove();
      gNodes.selectAll("*").remove();
      const data = sampleTrees[idx];
      if (!data) return;

      // Build hierarchy and layout
      const root = d3.hierarchy(data, d => d.children);
      treeLayout(root);

      // LINKS: red if both ends active, grey otherwise
      gLinks.selectAll("path")
        .data(root.links())
        .join("path")
          .attr("class", d =>
            `link ${(d.source.data.active && d.target.data.active) ? 'active' : 'default'}`
          )
          .attr("d", d3.linkHorizontal()
            .x(d => d.y + 80)
            .y(d => d.x + 10)
          );

      // NODES
      const node = gNodes.selectAll("g.node")
        .data(root.descendants())
        .join("g")
          .attr("class", "node")
          .attr("transform", d => `translate(${d.y+80},${d.x+10})`);

      node.append("circle")
          .attr("r", 6)
          .on("click", (event, d) => showDetails(d.data));

      node.append("text")
          .attr("dy", 3)
          .attr("x", d => d.children ? -12 : 12)
          .style("text-anchor", d => d.children ? "end" : "start")
          .text(d => {
            if (!d.parent) {
              // root: show the target question
              return d.data.Target?.[0]?.Name || "";
            } else {
              // non-root: newly bound var
              const me = d.data.Bound_cond || [];
              const pa = d.parent.data.Bound_cond || [];
              for (let v of me) {
                if (!pa.find(pv =>
                    pv.Name === v.Name &&
                    JSON.stringify(pv.Value) === JSON.stringify(v.Value)
                )) {
                  return `${v.Name} = [${v.Value.join(", ")}]`;
                }
              }
              return "";
            }
          });
    }

    function showDetails(node) {
      // drop children, render colored sections
      const { children, ...n } = node;
      let html = "";

      // Probability
      html += `<div style="background:#fffae6;padding:8px;margin-bottom:6px;border-radius:4px;">
                 <strong>Probability:</strong> ${n.prob}
               </div>`;

      // Target
      if (n.Target) {
        html += `<div style="background:#e6f3ff;padding:8px;margin-bottom:6px;border-radius:4px;">
                   <strong>Target</strong>`;
        n.Target.forEach(v => {
          html += `<div>• ${v.Name}: [${v.Value.join(", ")}]</div>`;
        });
        html += `</div>`;
      }

      // Unbound
      if (n.Unbound_cond) {
        html += `<div style="background:#fffbe6;padding:8px;margin-bottom:6px;border-radius:4px;">
                   <strong>Unbound</strong>`;
        n.Unbound_cond.forEach(v => {
          const p = v.Prob ? ` P=[${v.Prob.join(", ")}]` : "";
          html += `<div>• ${v.Name}: [${v.Value.join(", ")}]${p}</div>`;
        });
        html += `</div>`;
      }

      // Bound
      if (n.Bound_cond) {
        html += `<div style="background:#e6ffe6;padding:8px;margin-bottom:6px;border-radius:4px;">
                   <strong>Bound</strong>`;
        n.Bound_cond.forEach(v => {
          html += `<div>• ${v.Name}: [${v.Value.join(", ")}]</div>`;
        });
        html += `</div>`;
      }

      document.getElementById("node-json").innerHTML = html;
    }
  </script>
</body>
</html>
"""

def generate_html(log_file, output_html):
    sample_trees    = parse_sample_trees(log_file)
    final_prob      = parse_final_prob(log_file)
    if not sample_trees:
        print("✖ No sampling entries found in log.")
        return

    max_idx         = max(sample_trees.keys())
    sample_trees_json = json.dumps(sample_trees, indent=2)
    final_prob_json   = json.dumps(final_prob if final_prob is not None else [], indent=2)

    rendered = Template(HTML_TEMPLATE).render(
        sample_trees_json = sample_trees_json,
        final_prob_json   = final_prob_json,
        max_idx           = max_idx
    )

    with open(output_html, "w", encoding="utf-8") as f:
        f.write(rendered)
    print(f"✔ Generated HTML viewer: {output_html}")

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Generate D3‐based HTML viewer for sampling tree log with both active‐branch highlighting and final‐prob display."
    )
    p.add_argument("log", help="Path to the agent log file.")
    p.add_argument("out", help="Output HTML file path.")
    args = p.parse_args()

    if not os.path.exists(args.log):
        print(f"✖ Log file not found: {args.log}")
        exit(1)
    generate_html(args.log, args.out)
