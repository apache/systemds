# -------------------------------------------------------------
#
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
#
# -------------------------------------------------------------

import sys
import re
import networkx as nx
import matplotlib.pyplot as plt

try:
    import pygraphviz
    from networkx.drawing.nx_agraph import graphviz_layout
    HAS_PYGRAPHVIZ = True
except ImportError:
    HAS_PYGRAPHVIZ = False
    print("[WARNING] pygraphviz not found. Please install via 'pip install pygraphviz'.\n"
          "If not installed, we will use an alternative layout (spring_layout).")


def parse_line(line: str):
    """
    Parse a single line from the trace file to extract:
      - Node ID
      - Operation (hop name)
      - Kind (e.g., FOUT, LOUT, NREF)
      - Total cost
      - Weight
      - Refs (list of IDs that this node depends on)
    """

    # 1) Match a node ID in the form of "(R)" or "(<number>)"
    match_id = re.match(r'^\((R|\d+)\)', line)
    if not match_id:
        return None
    node_id = match_id.group(1)

    # 2) The remaining string after the node ID
    after_id = line[match_id.end():].strip()

    # Extract operation (hop name) before the first "["
    match_label = re.search(r'^(.*?)\s*\[', after_id)
    if match_label:
        operation = match_label.group(1).strip()
    else:
        operation = after_id.strip()

    # 3) Extract the kind (content inside the first pair of brackets "[]")
    match_bracket = re.search(r'\[([^\]]+)\]', after_id)
    if match_bracket:
        kind = match_bracket.group(1).strip()
    else:
        kind = ""

    # 4) Extract total and weight from the content inside curly braces "{}"
    total = ""
    weight = ""
    match_curly = re.search(r'\{([^}]+)\}', line)
    if match_curly:
        curly_content = match_curly.group(1)
        m_total = re.search(r'Total:\s*([\d\.]+)', curly_content)
        m_weight = re.search(r'Weight:\s*([\d\.]+)', curly_content)
        if m_total:
            total = m_total.group(1)
        if m_weight:
            weight = m_weight.group(1)

    # 5) Extract reference nodes: look for the first parenthesis containing numbers after the hop name
    match_refs = re.search(r'\(\s*(\d+(?:,\d+)*)\s*\)', after_id)
    if match_refs:
        ref_str = match_refs.group(1)
        refs = [r.strip() for r in ref_str.split(',') if r.strip().isdigit()]
    else:
        refs = []

    return {
        'node_id': node_id,
        'operation': operation,
        'kind': kind,
        'total': total,
        'weight': weight,
        'refs': refs
    }


def build_dag_from_file(filename: str):
    """
    Read a trace file line by line and build a directed acyclic graph (DAG) using NetworkX.
    """
    G = nx.DiGraph()
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            info = parse_line(line)
            if not info:
                continue

            node_id = info['node_id']
            operation = info['operation']
            kind = info['kind']
            total = info['total']
            weight = info['weight']
            refs = info['refs']

            # Add node with attributes
            G.add_node(node_id, label=operation, kind=kind, total=total, weight=weight)

            # Add edges from references to this node
            for r in refs:
                if r not in G:
                    G.add_node(r, label=r, kind="", total="", weight="")
                G.add_edge(r, node_id)
    return G


def main():
    """
    Main function that:
      - Reads a filename from command-line arguments
      - Builds a DAG from the file
      - Draws and displays the DAG using matplotlib
    """

    # Get filename from command-line argument
    if len(sys.argv) < 2:
        print("[ERROR] No filename provided.\nUsage: python plot_federated_dag.py <filename>")
        sys.exit(1)
    filename = sys.argv[1]

    print(f"[INFO] Running with filename '{filename}'")

    # Build the DAG
    G = build_dag_from_file(filename)

    # Print debug info: nodes and edges
    print("Nodes:", G.nodes(data=True))
    print("Edges:", list(G.edges()))

    # Decide on layout
    if HAS_PYGRAPHVIZ:
        # graphviz_layout with rankdir=BT (bottom to top), etc.
        pos = graphviz_layout(G, prog='dot', args='-Grankdir=BT -Gnodesep=0.5 -Granksep=0.8')
    else:
        # Fallback layout if pygraphviz is not installed
        pos = nx.spring_layout(G, seed=42)

    # Dynamically adjust figure size based on number of nodes
    node_count = len(G.nodes())
    fig_width = 10 + node_count / 10.0
    fig_height = 6 + node_count / 10.0
    plt.figure(figsize=(fig_width, fig_height), facecolor='white', dpi=300)
    ax = plt.gca()
    ax.set_facecolor('white')

    # Generate labels for each node in the format:
    # node_id: operation_name
    # C<total> (W<weight>)
    labels = {
        n: f"{n}: {G.nodes[n].get('label', n)}\n C{G.nodes[n].get('total', '')} (W{G.nodes[n].get('weight', '')})"
        for n in G.nodes()
    }

    # Function to determine color based on 'kind'
    def get_color(n):
        k = G.nodes[n].get('kind', '').lower()
        if k == 'fout':
            return 'tomato'
        elif k == 'lout':
            return 'dodgerblue'
        elif k == 'nref':
            return 'mediumpurple'
        else:
            return 'mediumseagreen'

    # Determine node shapes based on operation name:
    #  - '^' (triangle) if the label contains "twrite"
    #  - 's' (square) if the label contains "tread"
    #  - 'o' (circle) otherwise
    triangle_nodes = [n for n in G.nodes() if 'twrite' in G.nodes[n].get('label', '').lower()]
    square_nodes = [n for n in G.nodes() if 'tread' in G.nodes[n].get('label', '').lower()]
    other_nodes = [
        n for n in G.nodes()
        if 'twrite' not in G.nodes[n].get('label', '').lower() and
           'tread' not in G.nodes[n].get('label', '').lower()
    ]

    # Colors for each group
    triangle_colors = [get_color(n) for n in triangle_nodes]
    square_colors = [get_color(n) for n in square_nodes]
    other_colors = [get_color(n) for n in other_nodes]

    # Draw nodes group-wise
    node_collection_triangle = nx.draw_networkx_nodes(
        G, pos, nodelist=triangle_nodes, node_size=800,
        node_color=triangle_colors, node_shape='^', ax=ax
    )
    node_collection_square = nx.draw_networkx_nodes(
        G, pos, nodelist=square_nodes, node_size=800,
        node_color=square_colors, node_shape='s', ax=ax
    )
    node_collection_other = nx.draw_networkx_nodes(
        G, pos, nodelist=other_nodes, node_size=800,
        node_color=other_colors, node_shape='o', ax=ax
    )

    # Set z-order for nodes, edges, and labels
    node_collection_triangle.set_zorder(1)
    node_collection_square.set_zorder(1)
    node_collection_other.set_zorder(1)

    edge_collection = nx.draw_networkx_edges(G, pos, arrows=True, arrowstyle='->', ax=ax)
    if isinstance(edge_collection, list):
        for ec in edge_collection:
            ec.set_zorder(2)
    else:
        edge_collection.set_zorder(2)

    label_dict = nx.draw_networkx_labels(G, pos, labels=labels, font_size=9, ax=ax)
    for text in label_dict.values():
        text.set_zorder(3)

    # Set the title
    plt.title("Program Level Federated Plan", fontsize=14, fontweight="bold")

    # Provide a small legend on the top-right or top-left
    plt.text(1, 1,
             "[LABEL]\n hopID: hopName\n C(Total) (W(Weight))",
             fontsize=12, ha='right', va='top', transform=ax.transAxes)

    # Example mini-legend for different 'kind' values
    plt.scatter(0.05, 0.95, color='dodgerblue', s=200, transform=ax.transAxes)
    plt.scatter(0.18, 0.95, color='tomato', s=200, transform=ax.transAxes)
    plt.scatter(0.31, 0.95, color='mediumpurple', s=200, transform=ax.transAxes)

    plt.text(0.08, 0.95, "LOUT", fontsize=12, va='center', transform=ax.transAxes)
    plt.text(0.21, 0.95, "FOUT", fontsize=12, va='center', transform=ax.transAxes)
    plt.text(0.34, 0.95, "NREF", fontsize=12, va='center', transform=ax.transAxes)

    plt.axis("off")

    # Save the plot to a file with the same name as the input file, but with a .png extension
    output_filename = f"{filename.rsplit('.', 1)[0]}.png"
    plt.savefig(output_filename, format='png', dpi=300, bbox_inches='tight')

    plt.show()


if __name__ == '__main__':
    main()
