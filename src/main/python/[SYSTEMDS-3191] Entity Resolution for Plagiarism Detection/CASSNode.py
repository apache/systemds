class CassNode:
    def __init__(self, label):

        self.label = str(label)
        self.children = []
        self.prevUse = -1 # keep track of the prev node usage
        self.nextUse = -1

        self.source_range = None

    def add_child(self, child):
        self.children.append(child)

    # Creating the Cass string
    def to_cass_string(self) -> str:

        cass_strings = []  # Store each function separately
        node_counter = {"current_id": 1}  # Start numbering at 1 for each function
    
        def traverse(node):
            if node.label == "removed":
                return "".join(traverse(child) for child in node.children)
            
            child_strings = [traverse(c) for c in node.children]

            if node.label.startswith("v"):
                return (f"{node.label}\t{node.prevUse}\t{node.nextUse}\t" + "".join(child_strings))
            
            if node.label.startswith("V") or node.label.startswith(("N", "C", "S", "F")):
                return f"{node.label}\t" + "".join(child_strings)

            child_count = len(node.children)
            return f"{node.label}\t{child_count}\t" + "".join(child_strings)

    # Process each top-level function separately
        for function_tree in self.children:
            node_counter["current_id"] = 1  # Reset numbering for each function
            cass_string = traverse(function_tree)
            cass_strings.append(cass_string)

        return cass_strings  # Return list of separate CASS strings
    
    def get_source_range_string(self):
        """
        Return the source-range string in the format "0,0,5,1"
        only if this node is a top-level S#FS function and
        source_range is set. Otherwise return a default "0,0,0,0".
        """
        if self.label.startswith("S#FS") and self.source_range is not None:
            (start_l, start_c, end_l, end_c) = self.source_range
            return f"{start_l},{start_c},{end_l},{end_c}"
        else:
            # Not S#FS or we don't have the source_range info
            return "0,0,0,0"
            
    
        
    # Getting all nodes in a Cass tree
    def get_node_count(self) -> int:
        
        node_counter = {"current_id": 1}  # Start numbering at 1
        total_nodes = 0

        def traverse(node):
            nonlocal total_nodes
            current_id = node_counter["current_id"]

            if node.label != "removed":
                node_counter["current_id"] += 1
                total_nodes += 1

            for child in node.children:
                traverse(child)


        traverse(self)

        return total_nodes
    
    # Generating a GraphViz DOT file with nodes numbered in the CASS-style creation order
    def to_dot(self):
        
        lines = ["digraph CASS {", "  node [shape=ellipse];"]
        node_counter = {"current_id": 1}  # Start numbering at 1
        edges = []
        node_map = {}  # Keep track of nodes by ID

        def traverse(node, parent_id=None):
            # Do not assign an ID or add the "removed" node
            if node.label == "removed":
                for child in node.children:
                    traverse(child, parent_id)  # Attach children directly to parent
                return

            current_id = node_counter["current_id"]
            node_counter["current_id"] += 1

            # Escape double quotes in label if necessary
            safe_label = node.label.replace('"', '\\"')
            node_map[current_id] = node.label  # Store node label

            lines.append(f'  n{current_id} [label="[{current_id}]: {safe_label}"];')

            # Create an edge from parent to the current node
            if parent_id is not None:
                edges.append(f'  n{parent_id} -> n{current_id};')

            # Visit children in the order they were added
            for child in node.children:
                traverse(child, current_id)

            return current_id

        # **Step 1: Traverse children of "removed" root, skipping "removed" itself**
        for child in self.children:
            traverse(child)

        # **Step 2: Add remaining edges to the DOT file**
        lines.extend(edges)

        # **Step 3: Close the DOT graph**
        lines.append("}")

        return lines
        

"""
    1) Numbering the nodes according to preorder Depth First Search Algorithm.
    2) For each node referencing a local variable (label 'vX'), record it in usage_map.
    3) After collecting, fill in .prevUse and .nextUse.
"""   

def assign_usage_links(root: CassNode):
    
    usage_map = {}

    current_id = [0] 

    def dfs(node: CassNode):
        
        if (node.label != "removed"):
            this_index = current_id[0]
            current_id[0] += 1

            # If it's a local variable usage, store in usage_map
            if node.label.startswith("v"):
                var_name = node.label[1:]  # "vsum" => "sum"
                if var_name not in usage_map:
                    usage_map[var_name] = []
                usage_map[var_name].append((this_index, node))

            # Recurse on children
            for child in node.children:
                dfs(child)
        else:
            
            for child in node.children:
                dfs(child)

    # Collect usage in a DFS
    dfs(root)

    # For each variable, link up usage
    for var_name, usage_list in usage_map.items():
        # usage_list is e.g. [(4, nodeObj), (8, nodeObj), (21, nodeObj)]
        for i, (this_idx, node_obj) in enumerate(usage_list):
            # prev
            if i > 0:
                prev_idx = usage_list[i-1][0]
                node_obj.prevUse = prev_idx
            else:
                node_obj.prevUse = -1

            # next
            if i < len(usage_list)-1:
                next_idx = usage_list[i+1][0]
                node_obj.nextUse = next_idx
            else:
                node_obj.nextUse = -1