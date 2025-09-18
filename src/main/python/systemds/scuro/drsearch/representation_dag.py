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
from dataclasses import dataclass
from typing import List, Dict, Any
from systemds.scuro.modality.modality import Modality
from systemds.scuro.modality.transformed import TransformedModality
from systemds.scuro.representations.representation import (
    Representation as UnimodalRepresentation,
)
from systemds.scuro.representations.aggregated_representation import (
    AggregatedRepresentation,
)
from systemds.scuro.representations.context import Context


@dataclass
class RepresentationDag:

    def __init__(self, nodes: List[Any], root_node_id):
        self.root_node_id = root_node_id
        self.nodes = self.filter_connected_nodes(nodes)

    def filter_connected_nodes(self, nodes):
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

    def get_node_by_id(self, node_id: str):
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

    def execute(self, modalities: List[Modality]) -> Dict[str, TransformedModality]:
        cache = {}

        def execute_node(node_id: str) -> TransformedModality:
            if node_id in cache:
                return cache[node_id]

            node = self.get_node_by_id(node_id)

            if not node.inputs:
                if hasattr(node, "representation_index"):
                    modality = get_modality_by_id_and_instance_id(
                        modalities, node.modality_id, node.representation_index
                    )
                else:
                    modality = get_modality_by_id(modalities, node.modality_id)
                cache[node_id] = modality
                return modality

            input_mods = [execute_node(input_id) for input_id in node.inputs]

            if len(input_mods) == 1:
                if isinstance(node.operation(), Context):
                    result = input_mods[0].context(node.operation())
                elif isinstance(node.operation(), UnimodalRepresentation):
                    if (
                        isinstance(input_mods[0], TransformedModality)
                        and input_mods[0].transformation[0].__class__ == node.operation
                    ):
                        result = input_mods[0]
                    else:
                        result = input_mods[0].apply_representation(node.operation())
                elif isinstance(node.operation(), AggregatedRepresentation):
                    result = node.operation().transform(input_mods[0])
            else:
                result = input_mods[0].combine(input_mods[1:], node.operation())

            cache[node_id] = result
            return result

        execute_node(self.root_node_id)

        return cache


def get_modality_by_id(modalities: List[Modality], modality_id: int) -> Modality:
    for modality in modalities:
        if modality.modality_id == modality_id:
            return modality
    return None


def get_modality_by_id_and_instance_id(
    modalities: List[Modality], modality_id: int, instance_id: int
):
    counter = 0
    for modality in modalities:
        if modality.modality_id == modality_id:
            if counter == instance_id:
                return modality
            else:
                counter += 1
    return None
