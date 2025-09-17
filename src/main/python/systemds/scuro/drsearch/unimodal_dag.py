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
from dataclasses import dataclass, field
from typing import List, Dict, Any
import copy
from collections import deque


@dataclass
class UnimodalNode:
    node_id: str
    operation: Any
    inputs: List[str]
    modality_id: str = None
    parameters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class UnimodalDAG:

    def __init__(self, nodes: List[UnimodalNode], root_node_id):
        self.root_node_id = root_node_id
        self.nodes = self.filter_connected_nodes_bfs(nodes)

    def filter_connected_nodes_bfs(self, nodes):
        node_map = {node.node_id: node for node in nodes}

        if self.root_node_id not in node_map:
            return []

        visited = set()
        stack = [self.root_node_id]

        while stack:
            current_id = stack.pop()
            if current_id not in visited:
                visited.add(current_id)

                current_node = node_map[current_id]
                for input_id in current_node.inputs:
                    if input_id in node_map and input_id not in visited:
                        stack.append(input_id)

        return [node for node in nodes if node.node_id in visited]

    def get_leaf_nodes(self) -> List[str]:
        leaf_nodes = []
        for node in self.nodes:
            if not node.inputs:
                leaf_nodes.append(node.node_id)
        return leaf_nodes

    def get_node_by_id(self, node_id: str) -> UnimodalNode:
        for node in self.nodes:
            if node.node_id == node_id:
                return node
        return None

    def get_children(self, node_id: str) -> List[str]:
        children = []
        for node in self.nodes:
            if node_id in node.inputs:
                children.append(node.node_id)
        return children

    def validate(self) -> bool:
        node_ids = {node.node_id for node in self.nodes}

        if self.root_node_id not in node_ids:
            return False

        for node in self.nodes:
            for input_id in node.inputs:
                if input_id not in node_ids:
                    return False

        visited = set()

        def has_cycle(node_id: str, path: set) -> bool:
            if node_id in path:
                return True
            if node_id in visited:
                return False
            path.add(node_id)
            visited.add(node_id)
            node = self.get_node_by_id(node_id)
            for input_id in node.inputs:
                if has_cycle(input_id, path.copy()):
                    return True
            return False

        return not has_cycle(self.root_node_id, set())


class UnimodalDAGBuilder:

    def __init__(self):
        self.nodes = []
        self.node_counter = 0

    def create_leaf_node(self, operation: Any, modality_id: str) -> str:
        node_id = f"leaf_{self.node_counter}"
        self.node_counter += 1
        node = UnimodalNode(
            node_id=node_id, operation=operation, inputs=[], modality_id=modality_id
        )
        self.nodes.append(node)
        return node_id

    def create_operation_node(
        self, operation: Any, inputs: List[str], parameters: Dict[str, Any] = None
    ) -> str:
        node_id = f"op_{self.node_counter}"
        self.node_counter += 1
        node = UnimodalNode(
            node_id=node_id,
            operation=operation,
            inputs=inputs,
            parameters=parameters or {},
        )
        self.nodes.append(node)
        return node_id

    def build(self, root_node_id: str) -> UnimodalDAG:
        dag = UnimodalDAG(nodes=copy.deepcopy(self.nodes), root_node_id=root_node_id)
        if not dag.validate():
            raise ValueError("Invalid DAG construction")
        return dag
