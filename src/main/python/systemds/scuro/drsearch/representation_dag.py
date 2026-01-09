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
import copy
from dataclasses import dataclass, field
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
from systemds.scuro.utils.identifier import get_op_id, get_node_id

from collections import OrderedDict
from typing import Any, Hashable, Optional


class LRUCache:
    def __init__(self, max_size: int = 256):
        self.max_size = max_size
        self._cache: "OrderedDict[Hashable, Any]" = OrderedDict()

    def get(self, key: Hashable) -> Optional[Any]:
        if key not in self._cache:
            return None
        value = self._cache.pop(key)
        self._cache[key] = value
        return value

    def put(self, key: Hashable, value: Any) -> None:
        if key in self._cache:
            self._cache.pop(key)
        elif len(self._cache) >= self.max_size:
            self._cache.popitem(last=False)
        self._cache[key] = value

    def __len__(self) -> int:
        return len(self._cache)


@dataclass
class RepresentationNode:
    node_id: str
    operation: Any
    inputs: List[str]
    modality_id: str = None
    representation_index: int = None
    parameters: Dict[str, Any] = field(default_factory=dict)


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

    def _compute_leaf_signature(self, node) -> Hashable:
        return ("leaf", node.modality_id, node.representation_index)

    def _compute_node_signature(self, node, input_sig_tuple) -> Hashable:
        op_cls = node.operation
        params_items = tuple(sorted((node.parameters or {}).items()))
        return ("op", op_cls, params_items, input_sig_tuple)

    def execute(
        self,
        modalities: List[Modality],
        task=None,
        external_cache: Optional[LRUCache] = None,
    ) -> Dict[str, TransformedModality]:
        cache: Dict[str, TransformedModality] = {}
        node_signatures: Dict[str, Hashable] = {}

        def execute_node(node_id: str, task) -> TransformedModality:
            if node_id in cache:
                return cache[node_id]

            node = self.get_node_by_id(node_id)

            if not node.inputs:
                modality = get_modality_by_id_and_instance_id(
                    modalities, node.modality_id, node.representation_index
                )
                cache[node_id] = modality
                node_signatures[node_id] = self._compute_leaf_signature(node)
                return modality

            input_mods = [execute_node(input_id, task) for input_id in node.inputs]
            input_signatures = tuple(
                node_signatures[input_id] for input_id in node.inputs
            )
            node_signature = self._compute_node_signature(node, input_signatures)
            is_unimodal = len(input_mods) == 1

            cached_result = None
            if external_cache and is_unimodal:
                cached_result = external_cache.get(node_signature)
            if cached_result is not None:
                result = cached_result

            else:
                node_operation = copy.deepcopy(node.operation())
                if len(input_mods) == 1:
                    # It's a unimodal operation
                    if isinstance(node_operation, Context):
                        result = input_mods[0].context(node_operation)
                    elif isinstance(node_operation, AggregatedRepresentation):
                        result = node_operation.transform(input_mods[0])
                    elif isinstance(node_operation, UnimodalRepresentation):
                        if (
                            isinstance(input_mods[0], TransformedModality)
                            and input_mods[0].transformation[0].__class__
                            == node.operation
                        ):
                            # Avoid duplicate transformations
                            result = input_mods[0]
                        else:
                            # Compute the representation
                            result = input_mods[0].apply_representation(node_operation)
                else:
                    # It's a fusion operation
                    fusion_op = node_operation
                    if (
                        hasattr(fusion_op, "needs_training")
                        and fusion_op.needs_training
                    ):
                        result = input_mods[0].combine_with_training(
                            input_mods[1:], fusion_op, task
                        )
                    else:
                        result = input_mods[0].combine(input_mods[1:], fusion_op)
                if external_cache and is_unimodal:
                    external_cache.put(node_signature, result)

            cache[node_id] = result
            node_signatures[node_id] = node_signature
            return result

        execute_node(self.root_node_id, task)

        return cache


def get_modality_by_id_and_instance_id(
    modalities: List[Modality], modality_id: int, instance_id: int
):
    counter = 0
    for modality in modalities:
        if modality.modality_id == modality_id:
            if counter == instance_id or instance_id == -1:
                return modality
            else:
                counter += 1
    return None


class RepresentationDAGBuilder:
    def __init__(self):
        self.nodes = []
        self.node_counter = 0

    def create_leaf_node(
        self, modality_id: str, representation_index: int = -1, operation=None
    ) -> str:
        if representation_index != -1:
            node_id = f"leaf_{modality_id}_{representation_index}"
        else:
            node_id = f"leaf_{get_node_id()}"
        node = RepresentationNode(
            node_id=node_id,
            inputs=[],
            operation=operation,
            modality_id=modality_id,
            representation_index=representation_index,
        )
        self.nodes.append(node)
        return node_id

    def create_operation_node(
        self, operation: Any, inputs: List[str], parameters: Dict[str, Any] = None
    ) -> str:
        node_id = f"op_{get_op_id()}"
        self.node_counter += 1
        node = RepresentationNode(
            node_id=node_id,
            inputs=inputs,
            operation=operation,
            parameters=parameters or {},
        )
        self.nodes.append(node)
        return node_id

    def build(self, root_node_id: str) -> RepresentationDag:
        dag = RepresentationDag(
            nodes=copy.deepcopy(self.nodes), root_node_id=root_node_id
        )
        if not dag.validate():
            raise ValueError("Invalid DAG construction")
        return dag

    def get_node(self, node_id: str) -> Optional[RepresentationNode]:
        for node in self.nodes:
            if node.node_id == node_id:
                return node
        return None
