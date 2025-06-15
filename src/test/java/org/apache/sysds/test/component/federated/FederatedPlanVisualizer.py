import re
import networkx as nx
import matplotlib.pyplot as plt
import os
import glob
import argparse
import sys

try:
    import pygraphviz
    from networkx.drawing.nx_agraph import graphviz_layout
    HAS_PYGRAPHVIZ = True
except ImportError:
    HAS_PYGRAPHVIZ = False
    print("[WARNING] pygraphviz not found. Please use 'pip install pygraphviz'.\n"
          "      If installation fails, alternative layouts like spring_layout will be used.")


# Operation and variable abbreviation dictionary
OPERATION_ABBR = {
    # General operators
    "TRead": "TR",
    "TWrite": "TW",
    "Aggregate": "Agg",
    "AggregateUnary": "AgU",
    "Binary": "Bin",
    "Unary": "Un",
    "Reorg": "Rog",
    "MatrixIndexing": "MIdx",
    "Transpose": "Trp",
    "Reshape": "Rshp",
    "Literal": "Lit",
    
    # Federation related operators
    "transferMatrix": "tMat",
    "transferMatrixFromRemoteToLocal": "t2Loc",
    "transferMatrixFromLocalToRemote": "t2Rem",
    "federated": "fed",
    "federatedOutput": "fOut",
    "localOutput": "lOut",
    "noderef": "nRef",
    
    # KMeans algorithm related operators
    "kmeans": "KM",
    "kmeansPredict": "KMP",
    "m_kmeans": "mKM",
    
    # Other operations
    "append": "app",
    "cbind": "cb",
    "rbind": "rb",
    "matrix": "mat",
    "conv2d": "c2d",
    "maxpool": "mxp",
    "convolution": "cnv",
    "pooling": "pool",
    "QuantizeMatrix": "QMat",
    "DeQuantizeMatrix": "DQMat"
}

# Variable abbreviation dictionary (commonly used variable names)
VARIABLE_ABBR = {
    "matrix": "Mat",
    "weight": "Wei",
    "input": "In",
    "output": "Out",
    "image": "Img",
    "prediction": "Pred",
    "target": "Tgt",
    "gradient": "Grad",
    "activation": "Act",
    "feature": "Feat",
    "label": "Lbl",
    "parameter": "Param",
    "temp": "Tmp",
    "temporary": "Tmp",
    "intermediate": "Imd",
    "result": "Res"
}

def parse_line(line: str):
    # Print original line
    print(f"Original line: {line}")
    
    # Skip empty lines or info lines like 'Additional Cost:'
    if not line or line.startswith("Additional Cost:"):
        return None
    
    # 1) Extract node ID
    match_id = re.match(r'^\((R|\d+)\)', line)
    if not match_id:
        print(f"  > Node ID not found: {line}")
        return None
    node_id = match_id.group(1)
    print(f"  > Node ID: {node_id}")

    # 2) Remaining string after node id
    after_id = line[match_id.end():].strip()
    print(f"  > String after ID: {after_id}")

    # hop name (label): string before the first "["
    match_label = re.search(r'^(.*?)\s*\[', after_id)
    if match_label:
        operation = match_label.group(1).strip()
    else:
        operation = after_id.strip()
    print(f"  > Hop name/operation: {operation}")

    # 3) kind: content inside the first brackets (e.g., "FOUT" or "LOUT")
    match_bracket = re.search(r'\[([^\]]+)\]', after_id)
    if match_bracket:
        kind = match_bracket.group(1).strip()
    else:
        kind = ""
    print(f"  > Kind: {kind}")

    # 4) total, self, weight: extract from content inside curly braces {}
    total = ""
    self_cost = ""
    weight = ""
    match_curly = re.search(r'\{([^}]+)\}', line)
    if match_curly:
        curly_content = match_curly.group(1)
        m_total = re.search(r'Total:\s*([\d\.]+)', curly_content)
        m_self = re.search(r'Self:\s*([\d\.]+)', curly_content)
        m_weight = re.search(r'Weight:\s*([\d\.]+)', curly_content)
        if m_total:
            total = m_total.group(1)
        if m_self:
            self_cost = m_self.group(1)
        if m_weight:
            weight = m_weight.group(1)
    print(f"  > Total: {total}, Self: {self_cost}, Weight: {weight}")

    # 5) Extract reference nodes (children): numbers inside the first parentheses after kind (multiple possible)
    child_ids = []
    # Find parentheses after the first [
    match_children = re.search(r'\[[^\]]+\]\s*\(([^)]+)\)', after_id)
    if match_children:
        children_str = match_children.group(1)
        print(f"  > Child node string: {children_str}")
        # Extract comma-separated IDs
        child_ids = [c.strip() for c in children_str.split(',') if c.strip()]
    print(f"  > Child Node IDs: {child_ids}")
    
    # 6) Edge details: extract from [Edges]{...}
    edge_details = {}
    match_edges = re.search(r'\[Edges\]\{(.*?)(?:\}|$)', line)
    if match_edges:
        edges_str = match_edges.group(1)
        print(f"  > [Edges] content: {edges_str}")
        
        # Separate each edge info by parentheses
        edge_items = re.findall(r'\(ID:[^)]+\)', edges_str)
        
        for item in edge_items:
            print(f"  > Part to parse: '{item}'")
            
            # Parse edge info: (ID:51, X, C:401810.0, F:0.0, FW:500.0)
            id_match = re.search(r'ID:(\d+)', item)
            xo_match = re.search(r',\s*([XO])', item)
            cumulative_match = re.search(r'C:([\d\.]+)', item)
            forward_match = re.search(r'F:([\d\.]+)', item)
            weight_match = re.search(r'FW:([\d\.]+)', item)
            
            if id_match:
                source_id = id_match.group(1)
                is_forwarding = xo_match and xo_match.group(1) == 'O'
                cumulative_cost = cumulative_match.group(1) if cumulative_match else None
                forward_cost = forward_match.group(1) if forward_match else "0.0"
                forward_weight = weight_match.group(1) if weight_match else "1.0"
                
                print(f"  > Parse edge details: source={source_id}, forwarding={'O' if is_forwarding else 'X'}, cumulative={cumulative_cost}, cost={forward_cost}, weight={forward_weight}")
                
                edge_details[source_id] = {
                    'is_forwarding': is_forwarding,
                    'cumulative_cost': cumulative_cost,
                    'forward_cost': forward_cost,
                    'forward_weight': forward_weight
                }

    print(f"  > Edge details: {edge_details}")
    print("-------------------------------------")

    return {
        'node_id': node_id,
        'operation': operation,
        'kind': kind,
        'total': total,
        'self_cost': self_cost,
        'weight': weight,
        'child_ids': child_ids,
        'edge_details': edge_details
    }


def build_dag_from_file(filename: str):
    G = nx.DiGraph()
    print(f"\n[INFO] Building graph from file '{filename}'.")
    
    line_count = 0
    parsed_count = 0
    
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            line_count += 1
            line = line.strip()
            if not line:
                continue

            info = parse_line(line)
            if not info:
                continue
                
            parsed_count += 1
            node_id = info['node_id']
            operation = info['operation']
            kind = info['kind']
            total = info['total']
            self_cost = info['self_cost']
            weight = info['weight']
            child_ids = info['child_ids']
            edge_details = info['edge_details']

            print(f"Adding node: {node_id}, label: {operation}, kind: {kind}")
            G.add_node(node_id, label=operation, kind=kind, total=total, self_cost=self_cost, weight=weight)

            # 1. First create basic edges with child IDs in ()
            for child_id in child_ids:
                # Create child node if it doesn't exist
                if child_id not in G:
                    print(f"  > Creating missing child node: {child_id}")
                    G.add_node(child_id, label=child_id, kind="", total="", self_cost="", weight="")
                
                # Add edge from child node to current node (child -> parent)
                # Set as default (undiscovered edges marked with -1)
                print(f"  > Adding basic edge: {child_id} -> {node_id} (undiscovered edge)")
                G.add_edge(child_id, node_id, 
                          is_forwarding=False,
                          forward_cost="-1",  # Undiscovered edges marked with -1
                          forward_weight="-1",  # Undiscovered edges marked with -1
                          is_discovered=False)  # Additional flag
            
            # 2. Update edge attributes with [Edges] info
            for source_id, edge_data in edge_details.items():
                # Create source node if it doesn't exist
                if source_id not in G:
                    print(f"  > Creating missing source node: {source_id}")
                    G.add_node(source_id, label=source_id, kind="", total="", self_cost="", weight="")
                
                # Create edge if it doesn't exist, otherwise just update attributes
                if not G.has_edge(source_id, node_id):
                    # Set edge attributes
                    edge_attrs = {
                        'is_forwarding': edge_data['is_forwarding'],
                        'forward_cost': edge_data['forward_cost'],
                        'forward_weight': edge_data['forward_weight'],
                        'is_discovered': True  # Edge discovered in [Edges]
                    }
                    
                    # Add cumulative cost if available
                    if 'cumulative_cost' in edge_data and edge_data['cumulative_cost'] is not None:
                        edge_attrs['cumulative_cost'] = edge_data['cumulative_cost']
                        
                    print(f"  > Adding edge: {source_id} -> {node_id}, Forwarding: {edge_data['is_forwarding']}, Cost: {edge_data['forward_cost']}, Weight: {edge_data['forward_weight']}, Cumulative: {edge_data['cumulative_cost']}")
                    G.add_edge(source_id, node_id, **edge_attrs)
                else:
                    print(f"  > Updating edge attributes: {source_id} -> {node_id}, Forwarding: {edge_data['is_forwarding']}, Cost: {edge_data['forward_cost']}, Weight: {edge_data['forward_weight']}, Cumulative: {edge_data['cumulative_cost']}")
                    G[source_id][node_id]['is_forwarding'] = edge_data['is_forwarding']
                    G[source_id][node_id]['forward_cost'] = edge_data['forward_cost']
                    G[source_id][node_id]['forward_weight'] = edge_data['forward_weight']
                    G[source_id][node_id]['is_discovered'] = True  # Edge discovered in Edges
                    
                    # Add cumulative cost if available
                    if 'cumulative_cost' in edge_data and edge_data['cumulative_cost'] is not None:
                        G[source_id][node_id]['cumulative_cost'] = edge_data['cumulative_cost']

    print(f"\n[INFO] Parsed {parsed_count} nodes out of {line_count} total lines.")
    print(f"[INFO] Graph info: {len(G.nodes())} nodes, {len(G.edges())} edges\n")
    
    print("--- Node Information ---")
    for node, data in G.nodes(data=True):
        print(f"Node {node}: {data}")
    
    print("\n--- Edge Information ---")
    for u, v, data in G.edges(data=True):
        print(f"Edge {u} -> {v}: {data}")
    
    return G


def get_unique_filename(base_filename: str) -> str:
    """Generate new filename by incrementing if existing file exists"""
    if not os.path.exists(base_filename):
        return base_filename
    
    name, ext = os.path.splitext(base_filename)
    counter = 1
    while True:
        new_filename = f"{name}_{counter}{ext}"
        if not os.path.exists(new_filename):
            return new_filename
        counter += 1


def format_number(num_str):
    """Format numbers as strings. Numbers with 3 or more digits are converted to mathematical exponential notation."""
    try:
        num = float(num_str)
        if num >= 1000 or num <= -1000:
            # Calculate exponent
            exponent = 0
            base = abs(num)
            while base >= 10:
                base /= 10
                exponent += 1
            
            sign = "-" if num < 0 else ""
            # Round to first decimal place
            base_rounded = round(base, 1)
            base_str = f"{sign}{base_rounded}"
            
            # Convert exponent to Unicode superscript
            superscript_map = {
                '0': '⁰', '1': '¹', '2': '²', '3': '³', '4': '⁴',
                '5': '⁵', '6': '⁶', '7': '⁷', '8': '⁸', '9': '⁹',
                '+': '⁺', '-': '⁻'
            }
            
            exp_str = str(exponent)
            superscript_exp = ''.join(superscript_map[c] for c in exp_str)
            
            return f"{base_str}×10{superscript_exp}"
        else:
            # Round to first decimal place
            rounded_num = round(num, 1)
            # If integer after rounding, display as integer; otherwise display to first decimal place
            if rounded_num == int(rounded_num):
                return str(int(rounded_num))
            else:
                return str(rounded_num)
    except (ValueError, TypeError):
        return str(num_str)


def get_abbreviated_label(label):
    """
    Abbreviate labels using abbreviation dictionary.
    Example: "transferMatrixFromRemoteToLocal" -> "t2Loc"
    """
    if not label:
        return label
    
    # Split label words (by CamelCase, snake_case, spaces, etc.)
    # 1. CamelCase -> spaced
    spaced_label = re.sub(r'([a-z])([A-Z])', r'\1 \2', label)
    # 2. snake_case -> spaced
    spaced_label = spaced_label.replace('_', ' ')
    # 3. Split by spaces
    words = spaced_label.split()
    
    result = []
    for word in words:
        # Check operator abbreviation
        if (word.lower() == "op"):
            continue

        is_abbreviated = False
        for op, abbr in OPERATION_ABBR.items():
            if op.lower() == word.lower():
                result.append(abbr)
                is_abbreviated = True
                break
        # Check variable abbreviation
        if not is_abbreviated:
            for var, abbr in VARIABLE_ABBR.items():
                if var.lower() == word.lower():
                    result.append(abbr)
                    break

        if not is_abbreviated:
            result.append(word)                 
                
    # Connect words using separator character (·)
    abbreviated = '·'.join(result)
    abbreviated = truncate_label(abbreviated)

    return abbreviated


def truncate_label(label, max_length=8):
    """Limit label name to specified maximum length."""
    if not label or len(label) <= max_length:
        return label
    return label[:max_length-1]


def visualize_plan(filename: str, output_dir: str = "visualization_output", 
                node_cost_display: bool = True, edge_cost_display: bool = True):
    print(f"[INFO] Visualizing file '{filename}'.")
    print(f"[INFO] Node cost display: {'Enabled' if node_cost_display else 'Disabled'}")
    print(f"[INFO] Edge cost display: {'Enabled' if edge_cost_display else 'Disabled'}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    G = build_dag_from_file(filename)
    print("Nodes:", G.nodes(data=True))
    print("Edges:", list(G.edges(data=True)))

    if HAS_PYGRAPHVIZ:
        # Set larger node spacing (nodesep: horizontal spacing between nodes, ranksep: vertical spacing between levels)
        pos = graphviz_layout(G, prog='dot', args='-Grankdir=BT -Gnodesep=3 -Granksep=3')
    else:
        # For spring_layout, increase k value to ensure spacing between nodes
        pos = nx.spring_layout(G, seed=42, k=2.0)

    # Dynamically adjust overall graph size based on number of nodes
    node_count = len(G.nodes())
    fig_width = 15 + node_count / 8.0  # Increase width
    fig_height = 10 + node_count / 8.0  # Increase height
    plt.figure(figsize=(fig_width, fig_height), facecolor='white', dpi=300)
    ax = plt.gca()
    ax.set_facecolor('white')

    # Set node labels (format: id: hop name \n Total \n Self)
    labels = {}
    for n in G.nodes():
        # Basic information
        node_id = n
        label = G.nodes[n].get('label', n)
        total_cost = G.nodes[n].get('total', '')
        self_cost = G.nodes[n].get('self_cost', '')
        weight = G.nodes[n].get('weight', '')
        
        # Traverse child edges to calculate cumulative cost and forwarding cost totals
        child_cumulated_cost_sum = 0.0
        child_forward_cost_sum = 0.0
        
        print(f"\n[DEBUG] Calculating child costs for node {node_id}:")
        
        # 1. Find all edges coming into this node (child nodes)
        child_nodes = []
        for child, _, _ in G.in_edges(n, data=True):
            child_nodes.append(child)
        
        print(f"  Child nodes: {child_nodes}")
        
        # 2. Sum cumulative_cost and forward_cost for each child node
        for child_node in child_nodes:
            # Get edge data between current node and child node
            edge_data = G.get_edge_data(child_node, node_id)
            if edge_data:
                # Calculate cumulative cost
                if 'cumulative_cost' in edge_data and edge_data['cumulative_cost'] is not None:
                    try:
                        cumulative_cost = float(edge_data['cumulative_cost'])
                        print(f"  Cumulative cost for child node {child_node}: {cumulative_cost}")
                        child_cumulated_cost_sum += cumulative_cost
                    except ValueError:
                        print(f"  Failed to convert cumulative cost for child node {child_node}: {edge_data['cumulative_cost']}")
                
                # Calculate forwarding cost
                if 'forward_cost' in edge_data and edge_data['forward_cost'] is not None:
                    try:
                        if edge_data['forward_cost'] != '-1':  # Only for non-undiscovered edges
                            fwd_cost = float(edge_data['forward_cost'])
                            print(f"  Forward_cost for child node {child_node}: {fwd_cost}")
                            child_forward_cost_sum += fwd_cost
                    except ValueError:
                        print(f"  Failed to convert forward_cost for child node {child_node}: {edge_data['forward_cost']}")
        
        # First line of label: node ID, operation, total cost, weight
        first_line = f"{node_id}: {get_abbreviated_label(label)}"
        if node_cost_display:
            if total_cost:
                # Use format_number function instead of outputting only integer part
                formatted_total = format_number(total_cost)
                first_line += f"\nC: {formatted_total}"
            if weight:
                # Use format_number function instead of outputting only integer part
                formatted_weight = format_number(weight)
                first_line += f", W: {formatted_weight}"
            
            # Second line of label: Self Cost, child cumulative cost sum, child forwarding cost sum separated by slash (/)
            try:
                self_cost_formatted = format_number(self_cost) if self_cost else "0"
            except (ValueError, TypeError):
                self_cost_formatted = "0"
            
            child_cumulated_cost_formatted = format_number(child_cumulated_cost_sum)
            child_forward_cost_formatted = format_number(child_forward_cost_sum)
            
            print(f"  Final cost summary: Self={self_cost_formatted}, Child Total={child_cumulated_cost_formatted}, Child Fwd={child_forward_cost_formatted}")
            second_line = f"({self_cost_formatted}/{child_cumulated_cost_formatted}/{child_forward_cost_formatted})"
            
            # Final label
            labels[n] = f"{first_line}\n{second_line}"
        else:
            # Display only node ID and label without cost information
            labels[n] = first_line

    # Determine color for each node (based on kind)
    def get_color(n):
        k = G.nodes[n].get('kind', '').lower()
        if k == 'fout':
            return 'tomato'
        elif k == 'lout':
            return 'dodgerblue'
        elif k == 'nref':
            return 'mediumpurple'
        elif k == 'nref(top)':
            return 'darkviolet'
        else:
            return 'mediumseagreen'

    # Determine node shape (check if node's label contains specific strings):
    # If contains 'twrite' -> triangle (marker '^')
    # If contains 'tread' -> square (marker 's')
    # Otherwise -> circle (marker 'o')
    triangle_nodes = [n for n in G.nodes() if 'twrite' in G.nodes[n].get('label', '').lower()]
    square_nodes = [n for n in G.nodes() if 'tread' in G.nodes[n].get('label', '').lower()]
    other_nodes = [n for n in G.nodes() 
                   if 'twrite' not in G.nodes[n].get('label', '').lower() and
                      'tread' not in G.nodes[n].get('label', '').lower()]

    triangle_colors = [get_color(n) for n in triangle_nodes]
    square_colors = [get_color(n) for n in square_nodes]
    other_colors = [get_color(n) for n in other_nodes]

    # Increase node size
    node_size = 1200

    # Draw each node group separately
    node_collection_triangle = nx.draw_networkx_nodes(G, pos, nodelist=triangle_nodes, node_size=node_size, 
                                                      node_color=triangle_colors, node_shape='^', ax=ax)
    node_collection_square = nx.draw_networkx_nodes(G, pos, nodelist=square_nodes, node_size=node_size, 
                                                    node_color=square_colors, node_shape='s', ax=ax)
    node_collection_other = nx.draw_networkx_nodes(G, pos, nodelist=other_nodes, node_size=node_size, 
                                                   node_color=other_colors, node_shape='o', ax=ax)

    # Adjust zorder (nodes:1, edges:2, labels:3)
    node_collection_triangle.set_zorder(1)
    node_collection_square.set_zorder(1)
    node_collection_other.set_zorder(1)

    # Draw edges with different colors based on forwarding occurrence and ROOT node connection
    
    # 1. Normal edges (edges unrelated to ROOT node)
    normal_forwarding_edges = [(u, v) for u, v, d in G.edges(data=True) 
                              if 'is_discovered' in d and d['is_discovered'] 
                              and 'is_forwarding' in d and d['is_forwarding']
                              and v != 'R' and u != 'R']
    
    normal_non_forwarding_edges = [(u, v) for u, v, d in G.edges(data=True) 
                                  if 'is_discovered' in d and d['is_discovered'] 
                                  and 'is_forwarding' in d and not d['is_forwarding']
                                  and v != 'R' and u != 'R']
    
    # 2. All edges connected to ROOT node (both discovered/undiscovered shown in black)
    root_edges = [(u, v) for u, v, d in G.edges(data=True) 
                 if v == 'R' or u == 'R']
    
    # 3. Undiscovered edges (excluding those connected to ROOT node)
    undiscovered_edges = [(u, v) for u, v, d in G.edges(data=True) 
                         if ('is_discovered' not in d or not d['is_discovered'])
                         and v != 'R' and u != 'R']
    
    print(f"\n[DEBUG] Normal forwarding edges: {normal_forwarding_edges}")
    print(f"[DEBUG] Normal non-forwarding edges: {normal_non_forwarding_edges}")
    print(f"[DEBUG] ROOT connected edges: {root_edges}")
    print(f"[DEBUG] Undiscovered edges: {undiscovered_edges}")
    
    # Normal forwarding edges: red
    normal_forwarding_collection = nx.draw_networkx_edges(G, pos, edgelist=normal_forwarding_edges, 
                          arrows=True, arrowstyle='->', 
                          edge_color='red', width=2.0, ax=ax)
    
    # Normal non-forwarding edges: black
    normal_non_forwarding_collection = nx.draw_networkx_edges(G, pos, edgelist=normal_non_forwarding_edges, 
                          arrows=True, arrowstyle='->', 
                          edge_color='black', width=1.0, ax=ax)
    
    # All ROOT node connected edges: black
    root_edges_collection = nx.draw_networkx_edges(G, pos, edgelist=root_edges, 
                          arrows=True, arrowstyle='->', 
                          edge_color='black', width=1.0, ax=ax)
    
    # Undiscovered edges: purple thick line
    undiscovered_collection = nx.draw_networkx_edges(G, pos, edgelist=undiscovered_edges, 
                                                       arrows=True, arrowstyle='->', 
                                                       edge_color='purple', width=2.5, alpha=0.7, ax=ax)
    
    # Helper function for setting z-order
    def set_zorder_for_collection(collection, z=2):
        if isinstance(collection, list):
            for ec in collection:
                ec.set_zorder(z)
        elif collection is not None:
            collection.set_zorder(z)
    
    # Set z-order for all edge collections
    set_zorder_for_collection(normal_forwarding_collection)
    set_zorder_for_collection(normal_non_forwarding_collection)
    set_zorder_for_collection(root_edges_collection)
    set_zorder_for_collection(undiscovered_collection)

    # Add edge labels (forwarding cost and weight info) - set background completely transparent
    edge_labels = {}
    
    # Add edge labels only when edge_cost_display is True
    if edge_cost_display:
        # Display discovered edges in C/W/CC format (excluding ROOT node connections)
        for u, v, d in G.edges(data=True):
            # Don't display labels for edges connected to ROOT node
            if v == 'R' or u == 'R':
                continue
                
            # Display information for discovered edges
            if 'is_discovered' in d and d['is_discovered'] and 'forward_cost' in d and 'forward_weight' in d:
                label_parts = []

                # Add cumulative cost if available (integer part only)
                if 'cumulative_cost' in d and d['cumulative_cost'] is not None:
                    cumulative_cost_formatted = format_number(d['cumulative_cost'])
                    label_parts.append(f"C:{cumulative_cost_formatted}")

                # Forwarding cost 
                forward_cost_formatted = format_number(d['forward_cost'])
                label_parts.append(f"FC:{forward_cost_formatted}")
                
                # Weight
                forward_weight_formatted = format_number(d['forward_weight'])
                label_parts.append(f"FW:{forward_weight_formatted}")
                
                edge_labels[(u, v)] = "\n".join(label_parts)
            # Display undiscovered edges as "Undiscovered"
            elif ('is_discovered' not in d or not d['is_discovered']) and 'forward_cost' in d and 'forward_weight' in d:
                edge_labels[(u, v)] = "Undiscovered"

    # Add edge labels - set background completely transparent
    if edge_labels:
        edge_label_dict = nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, 
                                                     font_size=7, font_color='darkblue',
                                                     bbox=dict(boxstyle="round", fc="w", ec="none", alpha=0),
                                                     ax=ax)
        
        # Set label background directly transparent
        for key, text in edge_label_dict.items():
            text.set_bbox(dict(boxstyle="round", fc="none", ec="none", alpha=0))

    # Node labels - set background completely transparent
    label_dict = nx.draw_networkx_labels(G, pos, labels=labels, font_size=8, 
                                       bbox=dict(boxstyle="round", fc="w", ec="none", alpha=0),
                                       ax=ax)
    
    # Set node label background directly transparent
    for text in label_dict.values():
        text.set_zorder(3)
        text.set_bbox(dict(boxstyle="round", fc="none", ec="none", alpha=0))

    # Set desired title
    plt.title("Program Level Federated Plan", fontsize=16, fontweight="bold")

    # Node type legend (top left)
    plt.scatter(0.05, 0.95, color='dodgerblue', s=150, transform=ax.transAxes)
    plt.scatter(0.18, 0.95, color='tomato', s=150, transform=ax.transAxes)
    plt.scatter(0.31, 0.95, color='mediumpurple', s=150, transform=ax.transAxes)

    plt.text(0.08, 0.95, "LOUT", fontsize=10, va='center', transform=ax.transAxes)
    plt.text(0.21, 0.95, "FOUT", fontsize=10, va='center', transform=ax.transAxes)
    plt.text(0.34, 0.95, "NREF", fontsize=10, va='center', transform=ax.transAxes)
    
    # Edge related legend (top right)
    legend_x = 0.98  # Top right x coordinate
    legend_y = 0.98  # Top right y coordinate
    legend_spacing = 0.05  # Spacing between items
    
    # Label legend (text only)
    if node_cost_display:
        plt.text(legend_x, legend_y, "[Node LABEL]\nhopID: hopNam\nC: Total Cost, W: Weight\n(Self / Child Cum. Cost / Child Fwd. Cost)", 
                fontsize=12, ha='right', va='top', transform=ax.transAxes)
    else:
        plt.text(legend_x, legend_y, "[Node LABEL]\nhopID: hopNam", 
                fontsize=12, ha='right', va='top', transform=ax.transAxes)

    plt.axis("off")

    # Generate output filename based on input filename
    input_filename = os.path.basename(filename)
    base_output_filename = os.path.splitext(input_filename)[0]
    
    # Set filename suffix based on cost display options
    suffix = ""
    if not node_cost_display:
        suffix += "_no_node_cost"
    if not edge_cost_display:
        suffix += "_no_edge_cost"
    
    base_output_filename += suffix + ".png"
    output_filename = os.path.join(output_dir, base_output_filename)
    
    # Handle duplicate filenames
    output_filename = get_unique_filename(output_filename)
    
    plt.savefig(output_filename, bbox_inches='tight', dpi=300)
    print(f"[INFO] Visualization result saved to '{output_filename}'.")
    plt.close()


def main():
    
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Tool for visualizing federated plans')
    parser.add_argument('trace_file', help='Path to the trace file to visualize')
    parser.add_argument('--no-node-cost', action='store_true', help='Do not display node cost information')
    parser.add_argument('--no-edge-cost', action='store_true', help='Do not display edge cost information')
    parser.add_argument('--no-cost', action='store_true', help='Do not display any cost information (applies both --no-node-cost and --no-edge-cost)')
    parser.add_argument('--output-dir', default='visualization_output', help='Output directory path (default: visualization_output)')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Check file existence
    if not os.path.exists(args.trace_file):
        print(f"[ERROR] File '{args.trace_file}' not found.")
        sys.exit(1)
    
    # Set cost display options
    node_cost_display = not (args.no_node_cost or args.no_cost)
    edge_cost_display = not (args.no_edge_cost or args.no_cost)
    
    # Execute visualization
    visualize_plan(args.trace_file, args.output_dir, node_cost_display, edge_cost_display)


if __name__ == '__main__':
    main()
