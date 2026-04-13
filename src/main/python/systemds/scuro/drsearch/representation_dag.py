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
from typing import List, Dict, Set, Tuple, Union, Any, Hashable, Optional
from systemds.scuro.modality.modality import Modality
from systemds.scuro.modality.transformed import TransformedModality
from systemds.scuro.representations.representation import (
    Representation as UnimodalRepresentation,
)
from systemds.scuro.representations.aggregated_representation import (
    AggregatedRepresentation,
)
from systemds.scuro.representations.context import Context
from systemds.scuro.representations.dimensionality_reduction import (
    DimensionalityReduction,
)
from systemds.scuro.utils.identifier import get_op_id, get_node_id
from collections import OrderedDict, defaultdict, deque


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
    gpu_id: int = None


@dataclass
class RepresentationDag:

    def __init__(self, nodes: List[Any], root_node_id, dag_id: int = None):
        self.dag_id = dag_id
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

    def get_first_level_nodes(self) -> List[str]:
        leaf_nodes = set(self.get_leaf_nodes())
        first_level_nodes = []

        for node in self.nodes:
            if node.node_id in leaf_nodes:
                continue

            if node.inputs and all(input_id in leaf_nodes for input_id in node.inputs):
                first_level_nodes.append(node.node_id)

        return first_level_nodes

    def get_first_level_node_set(self) -> set:
        return set(self.get_first_level_nodes())

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

    def _get_node_representation_label(self, node: RepresentationNode) -> str:
        if not node.operation:
            if node.representation_index is None:
                return f"modality={node.modality_id}"
            return (
                f"modality={node.modality_id}, "
                f"representation_index={node.representation_index}"
            )

        try:
            operation_instance = node.operation(params=node.parameters)
            if hasattr(operation_instance, "name"):
                return str(operation_instance.name)
        except Exception:
            pass

        if hasattr(node.operation, "__name__"):
            return str(node.operation.__name__)
        return str(node.operation)

    def to_graphviz(self, graph_name: str = "RepresentationDag", rankdir: str = "BT"):
        try:
            from graphviz import Digraph
        except ImportError as exc:
            raise ImportError(
                "Graphviz visualization requires the 'graphviz' Python package. "
                "Install it with 'pip install graphviz' and ensure Graphviz "
                "binaries are available on your PATH."
            ) from exc

        graph = Digraph(name=graph_name)
        graph.attr(rankdir=rankdir)

        for node in self.nodes:
            node_label = f"{node.node_id}\\n{self._get_node_representation_label(node)}"
            node_shape = "ellipse" if not node.inputs else "box"
            node_attributes = {"shape": node_shape}
            if node.node_id == self.root_node_id:
                node_attributes["style"] = "bold"
            graph.node(node.node_id, label=node_label, **node_attributes)

        for node in self.nodes:
            for input_id in node.inputs:
                graph.edge(input_id, node.node_id)

        return graph

    def render_graphviz(
        self,
        output_path: str,
        format: str = "png",
        view: bool = False,
        cleanup: bool = True,
    ) -> str:
        graph = self.to_graphviz()
        return graph.render(
            filename=output_path, format=format, view=view, cleanup=cleanup
        )

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
        enable_cache=True,
        rep_cache: Dict[Any, TransformedModality] = None,
        consumer_count: Dict[str, int] = None,
        gpu_id: int = None,
    ) -> Union[Dict[str, TransformedModality], TransformedModality]:

        def execute_node(node_id: str, task) -> TransformedModality:
            if external_cache is not None:
                cached = external_cache.get(node_id)
                if cached is not None:
                    return cached

            node = self.get_node_by_id(node_id)

            if not node.inputs:
                modality = get_modality_by_id_and_instance_id(
                    modalities, int(node.modality_id), node.representation_index
                )

                return modality

            input_mods = [execute_node(input_id, task) for input_id in node.inputs]
            is_unimodal = len(input_mods) == 1

            if external_cache is not None and is_unimodal:
                cached = external_cache.get(node_id)
                if cached is not None:
                    result = cached
                else:
                    node_operation = node.operation(params=node.parameters)
                    if gpu_id is not None and hasattr(node_operation, "gpu_id"):
                        node_operation.gpu_id = gpu_id
                    if len(input_mods) == 1:
                        # It's a unimodal operation
                        if isinstance(node_operation, Context):
                            result = input_mods[0].context(node_operation)
                        elif isinstance(node_operation, DimensionalityReduction):
                            result = input_mods[0].dimensionality_reduction(
                                node_operation
                            )
                        elif isinstance(node_operation, AggregatedRepresentation):
                            result = node_operation.transform(input_mods[0])
                        elif isinstance(node_operation, UnimodalRepresentation):
                            if rep_cache is not None:
                                result = rep_cache[node_operation.name]
                            else:
                                # Compute the representation
                                result = input_mods[0].apply_representation(
                                    node_operation
                                )
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
            else:
                node_operation = node.operation(params=node.parameters)
                if gpu_id is not None and hasattr(node_operation, "gpu_id"):
                    node_operation.gpu_id = gpu_id
                if len(input_mods) == 1:
                    # It's a unimodal operation
                    if isinstance(node_operation, Context):
                        result = input_mods[0].context(node_operation)
                    elif isinstance(node_operation, DimensionalityReduction):
                        result = input_mods[0].dimensionality_reduction(node_operation)
                    elif isinstance(node_operation, AggregatedRepresentation):
                        result = node_operation.transform(input_mods[0])
                    elif isinstance(node_operation, UnimodalRepresentation):
                        if rep_cache is not None:
                            result = rep_cache[node_operation.name]
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

                if (
                    enable_cache
                    and external_cache is not None
                    and is_unimodal
                    and consumer_count[node_id] > 1
                ):
                    external_cache.put(node_id, result)
            return result

        result = execute_node(self.root_node_id, task)

        return result

    def compute_full_node_signature(
        self, node_id: str, node_signatures: Dict[str, Hashable] = None
    ) -> Hashable:
        if node_signatures is None:
            node_signatures = {}

        if node_id in node_signatures:
            return node_signatures[node_id]

        node = self.get_node_by_id(node_id)
        if not node:
            return None

        if not node.inputs:
            sig = self._compute_leaf_signature(node)
            node_signatures[node_id] = sig
            return sig

        input_sigs = []
        for input_id in sorted(node.inputs):
            input_sig = self.compute_full_node_signature(input_id, node_signatures)
            input_sigs.append(input_sig)

        sig = self._compute_node_signature(node, tuple(input_sigs))
        node_signatures[node_id] = sig
        return sig

    def find_nodes_with_same_predecessors(
        self, target_node_id: str, other_dag: "RepresentationDag", other_node_id: str
    ) -> bool:
        target_sig = self.compute_full_node_signature(target_node_id)
        other_sig = other_dag.compute_full_node_signature(other_node_id)
        return target_sig == other_sig

    def get_leaf_node_id(self) -> str:
        for node in self.nodes:
            if not node.inputs:
                return node.node_id
        return None


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
        self.dag_counter = 0

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

    def build(self, root_node_id: str, dag_id: int = None) -> RepresentationDag:
        dag = RepresentationDag(
            nodes=copy.deepcopy(self.nodes),
            root_node_id=root_node_id,
            dag_id=dag_id if dag_id is not None else self.dag_counter + 1,
        )
        self.dag_counter += 1
        if not dag.validate():
            raise ValueError("Invalid DAG construction")
        return dag

    def get_node(self, node_id: str) -> Optional[RepresentationNode]:
        for node in self.nodes:
            if node.node_id == node_id:
                return node
        return None


def get_consumer_count(dags: List[RepresentationDag]) -> Dict[str, int]:
    consumer_count: Dict[str, int] = defaultdict(int)
    for dag in dags:
        for node in dag.nodes:
            for inp in node.inputs:
                consumer_count[inp] += 1
    return consumer_count


def pushdown_aggregation(dag_group: List[RepresentationDag]) -> List[RepresentationDag]:
    consumer_count: Dict[str, int] = defaultdict(int)

    for dag in dag_group:
        for node in dag.nodes:
            for inp in node.inputs:
                consumer_count[inp] += 1

    processed_agg_ids: Set[str] = set()

    for dag in dag_group:
        agg_nodes = [
            n
            for n in dag.nodes
            if n.operation and issubclass(n.operation, AggregatedRepresentation)
        ]
        for agg_node in agg_nodes:
            if agg_node.node_id in processed_agg_ids:
                continue
            processed_agg_ids.add(agg_node.node_id)

            if len(agg_node.inputs) != 1:
                print(
                    f"Aggregation node {agg_node.node_id} has {len(agg_node.inputs)} inputs, skipping (SHOULD NOT HAPPEN)"
                )
                continue

            input_id = agg_node.inputs[0]

            processed_agg_ids.add(input_id)
            if consumer_count[input_id] != 1:
                continue

            input_node = None
            for d in dag_group:
                input_node = d.get_node_by_id(input_id)
                if input_node is not None:
                    break

            if not input_node or not input_node.operation:
                continue

            op_instance = input_node.operation(params=input_node.parameters)
            if op_instance.__class__.__bases__[0].__name__ != "BertFamily":
                continue

            input_node.parameters["_pushdown_aggregation"] = agg_node.parameters

            for d in dag_group:
                for node in d.nodes:
                    node.inputs = [
                        input_id if inp == agg_node.node_id else inp
                        for inp in node.inputs
                    ]

                if d.root_node_id == agg_node.node_id:
                    d.root_node_id = input_id

                d.nodes = [n for n in d.nodes if n.node_id != agg_node.node_id]

    return dag_group


class CSEAwareDAGBuilder:
    def __init__(self):
        self.global_nodes: List[RepresentationNode] = []
        self.signature_to_node: Dict[Hashable, str] = {}
        self.node_to_signature: Dict[str, Hashable] = {}
        self.node_counter = 0
        self.dag_counter = 0

    def _compute_node_signature(
        self, operation: Any, inputs: List[str], parameters: Dict[str, Any] = None
    ) -> Hashable:
        ip = [self.node_to_signature[inp] for inp in inputs]
        input_sigs = tuple(sorted(ip)) if inputs else ()
        op_cls = operation().name
        params_items = tuple(sorted((parameters or {}).items()))
        return ("op", op_cls, params_items, input_sigs)

    def _compute_leaf_signature(
        self, modality_id: str, representation_index: int = -1
    ) -> Hashable:
        return ("leaf", modality_id, representation_index)

    def _get_or_create_node(
        self,
        operation: Any,
        inputs: List[str],
        modality_id: str = None,
        representation_index: int = None,
        parameters: Dict[str, Any] = None,
        is_leaf: bool = False,
    ) -> str:
        if is_leaf:
            signature = self._compute_leaf_signature(modality_id, representation_index)
        else:
            signature = self._compute_node_signature(operation, inputs, parameters)

        try:
            if signature in self.signature_to_node:
                return self.signature_to_node[signature]
        except:
            pass

        if is_leaf:
            if representation_index != -1:
                node_id = f"leaf_{modality_id}_{representation_index}"
            else:
                node_id = f"leaf_{get_node_id()}"
        else:
            node_id = f"op_{get_op_id()}"
            self.node_counter += 1

        node = RepresentationNode(
            node_id=node_id,
            inputs=inputs,
            operation=operation,
            modality_id=modality_id,
            representation_index=representation_index,
            parameters=parameters or {},
            gpu_id=None,
        )

        self.global_nodes.append(node)
        self.signature_to_node[signature] = node_id
        self.node_to_signature[node_id] = signature

        return node_id

    def create_leaf_node(
        self, modality_id: str, representation_index: int = -1, operation=None
    ) -> str:
        return self._get_or_create_node(
            operation=operation,
            inputs=[],
            modality_id=modality_id,
            representation_index=representation_index,
            is_leaf=True,
        )

    def create_operation_node(
        self, operation: Any, inputs: List[str], parameters: Dict[str, Any] = None
    ):
        return self._get_or_create_node(
            operation=operation, inputs=inputs, parameters=parameters, is_leaf=False
        )

    def build(self, root_node_id: str, dag_id: int = None) -> RepresentationDag:
        dag = RepresentationDag(
            nodes=self.global_nodes,
            root_node_id=root_node_id,
            dag_id=dag_id if dag_id is not None else self.dag_counter + 1,
        )
        self.dag_counter += 1
        if not dag.validate():
            raise ValueError("Invalid DAG construction")
        return dag

    def get_node(self, node_id: str) -> Optional[RepresentationNode]:
        for node in self.global_nodes:
            if node.node_id == node_id:
                return node
        return None


def group_dags_by_dependencies(
    dags: List[RepresentationDag],
) -> List[List[RepresentationDag]]:
    if not dags:
        return []

    unique_dags: List[RepresentationDag] = []
    seen_signatures: set[Hashable] = set()

    for dag in dags:
        dag_sig = dag.compute_full_node_signature(dag.root_node_id)
        if dag_sig not in seen_signatures:
            seen_signatures.add(dag_sig)
            unique_dags.append(dag)

    dags = unique_dags
    dag_first_level_sets = []
    for dag in dags:
        first_level_nodes = dag.get_first_level_node_set()
        dag_first_level_sets.append(first_level_nodes)

    n = len(dags)
    parent = list(range(n))

    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(x, y):
        root_x = find(x)
        root_y = find(y)
        if root_x != root_y:
            parent[root_y] = root_x

    for i in range(n):
        for j in range(i + 1, n):
            if dag_first_level_sets[i] & dag_first_level_sets[j]:
                union(i, j)

    groups: Dict[int, List[RepresentationDag]] = {}
    for i in range(n):
        root = find(i)
        if root not in groups:
            groups[root] = []
        groups[root].append(dags[i])

    return list(groups.values())


def dags_to_graphviz(
    dags: List[RepresentationDag],
    graph_name: str = "RepresentationDagGroup",
    rankdir: str = "BT",
    show_dag_roots: bool = True,
):
    try:
        from graphviz import Digraph
    except ImportError as exc:
        raise ImportError(
            "Graphviz visualization requires the 'graphviz' Python package. "
            "Install it with 'pip install graphviz' and ensure Graphviz "
            "binaries are available on your PATH."
        ) from exc

    graph = Digraph(name=graph_name)
    graph.attr(rankdir=rankdir)

    root_ids = {dag.root_node_id for dag in dags}
    rendered_nodes: Set[str] = set()
    rendered_edges: Set[Tuple[str, str]] = set()

    for dag_idx, dag in enumerate(dags):
        if show_dag_roots:
            dag_node_id = f"dag_{dag_idx}"
            graph.node(
                dag_node_id,
                label=f"DAG {dag_idx}\\nroot={dag.root_node_id}",
                shape="oval",
                style="dashed",
            )
            graph.edge(dag_node_id, dag.root_node_id, style="dashed")

        for node in dag.nodes:
            if node.node_id not in rendered_nodes:
                node_label = (
                    f"{node.node_id}\\n{dag._get_node_representation_label(node)}"
                )
                node_shape = "ellipse" if not node.inputs else "box"
                node_attributes = {"shape": node_shape}
                if node.node_id in root_ids:
                    node_attributes["style"] = "bold"
                graph.node(node.node_id, label=node_label, **node_attributes)
                rendered_nodes.add(node.node_id)

            for input_id in node.inputs:
                edge_key = (input_id, node.node_id)
                if edge_key not in rendered_edges:
                    graph.edge(input_id, node.node_id)
                    rendered_edges.add(edge_key)

    return graph


def render_dags_graphviz(
    dags: List[RepresentationDag],
    output_path: str,
    format: str = "png",
    view: bool = False,
    cleanup: bool = True,
) -> str:
    graph = dags_to_graphviz(dags)
    return graph.render(filename=output_path, format=format, view=view, cleanup=cleanup)
