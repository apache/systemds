import itertools
import time
from dataclasses import dataclass, field
from typing import List, Dict, Any, Generator
import copy
import traceback

from systemds.scuro import Task
from systemds.scuro.representations.aggregated_representation import (
    AggregatedRepresentation,
)
from systemds.scuro.representations.aggregate import Aggregation
from systemds.scuro.drsearch.operator_registry import Registry
from systemds.scuro.utils.schema_helpers import get_shape
from systemds.scuro.modality.transformed import TransformedModality


@dataclass
class MultimodalNode:
    node_id: str
    inputs: List[str]
    operation: Any
    modality_id: str = None
    representation_index: int = None
    parameters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MultimodalDAG:
    def __init__(self, nodes: List[MultimodalNode], root_node_id: str):
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

    def get_node_by_id(self, node_id: str) -> MultimodalNode:
        for node in self.nodes:
            if node.node_id == node_id:
                return node
        return None

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

    def execute(
        self, k_best_representations: Dict[str, Dict[str, List[Any]]], task: Task
    ) -> TransformedModality:
        cache = {}

        def execute_node(node_id: str) -> TransformedModality:
            if node_id in cache:
                return cache[node_id]

            node = self.get_node_by_id(node_id)

            if not node.inputs:
                representation = k_best_representations[task.model.name][node.modality_id][
                    "representations"
                ][node.representation_index]
                cache[node_id] = representation
                return representation

            input_representations = [execute_node(input_id) for input_id in node.inputs]

            if len(input_representations) == 1:
                result = input_representations[0]
            else:
                op = node.operation
                try:
                    op_instance = op() if callable(op) else op
                except Exception:
                    op_instance = op

                result = input_representations[0].combine(
                    input_representations[1:], op_instance
                )

            cache[node_id] = result
            return result

        return execute_node(self.root_node_id)


class MultimodalDAGBuilder:
    def __init__(self):
        self.nodes = []
        self.node_counter = 0

    def create_leaf_node(self, modality_id: str, representation_index: int) -> str:
        node_id = f"leaf_{modality_id}_{representation_index}"
        node = MultimodalNode(
            node_id=node_id,
            inputs=[],
            operation=None,
            modality_id=modality_id,
            representation_index=representation_index,
        )
        self.nodes.append(node)
        return node_id

    def create_fusion_node(self, inputs: List[str], fusion_operation: Any) -> str:
        node_id = f"fusion_{self.node_counter}"
        self.node_counter += 1
        node = MultimodalNode(
            node_id=node_id,
            inputs=inputs,
            operation=fusion_operation,
            parameters=(
                fusion_operation.parameters
                if hasattr(fusion_operation, "parameters")
                else {}
            ),
        )
        self.nodes.append(node)
        return node_id

    def build(self, root_node_id: str) -> MultimodalDAG:
        dag = MultimodalDAG(nodes=copy.deepcopy(self.nodes), root_node_id=root_node_id)
        if not dag.validate():
            raise ValueError("Invalid DAG construction")
        return dag


class MultimodalOptimizer:
    def __init__(
        self,
        modalities: List[Any],
        unimodal_optimization_results: Any,
        tasks: List[Any],
        k: int = 2,
        debug: bool = True,
        min_modalities: int = 2,
        max_modalities: int = None,
    ):
        self.modalities = modalities
        self.unimodal_optimization_results = unimodal_optimization_results
        self.tasks = tasks
        self.k = k
        self.debug = debug
        self.min_modalities = max(2, min_modalities)
        self.max_modalities = max_modalities or len(modalities)

        self.operator_registry = Registry()
        self.fusion_operators = self.operator_registry.get_fusion_operators()

        self.k_best_representations = self._extract_k_best_representations()
        self.optimization_results = []

    def _extract_k_best_representations(self) -> Dict[str, Dict[str, List[Any]]]:
        k_best = {}

        for task in self.tasks:
            k_best[task.model.name] = {}

            for modality in self.modalities:
                k_best_results, cached_data = (
                    self.unimodal_optimization_results.get_k_best_results(
                        modality, self.k, task
                    )
                )

                k_best[task.model.name][modality.modality_id] = {
                    "results": k_best_results,
                    "representations": cached_data,
                }

        return k_best

    def _generate_modality_combinations(self) -> Generator[List[str], None, None]:
        modality_ids = [mod.modality_id for mod in self.modalities]

        for r in range(
            self.min_modalities, min(self.max_modalities + 1, len(modality_ids) + 1)
        ):
            for modality_subset in itertools.combinations(modality_ids, r):
                yield list(modality_subset)

    def _generate_representation_combinations(
        self, modality_subset: List[str], task_name: str
    ) -> Generator[Dict[str, int], None, None]:
        representation_options = []

        for modality_id in modality_subset:
            num_representations = len(
                self.k_best_representations[task_name][modality_id]["representations"]
            )
            representation_options.append(list(range(num_representations)))

        for combo in itertools.product(*representation_options):
            yield {
                modality_id: repr_idx
                for modality_id, repr_idx in zip(modality_subset, combo)
            }

    def _generate_fusion_dags(
        self, modality_subset: List[str], representation_combo: Dict[str, int]
    ) -> Generator[MultimodalDAG, None, None]:
        leaf_infos = [(m, representation_combo[m]) for m in modality_subset]

        def gen_trees(indices: List[int]):
            if len(indices) == 1:
                yield indices[0]
                return
            for split in range(1, len(indices)):
                for left_idxs in itertools.combinations(indices, split):
                    left = list(left_idxs)
                    right = [i for i in indices if i not in left]
                    for l_tree in gen_trees(left):
                        for r_tree in gen_trees(right):
                            yield (l_tree, r_tree)

        def build_variants(subtree, base_builder: MultimodalDAGBuilder, leaf_id_map):
            variants = []

            if isinstance(subtree, int):
                variants.append((base_builder, leaf_id_map[subtree]))
                return variants

            left_sub, right_sub = subtree

            left_variants = build_variants(left_sub, copy.deepcopy(base_builder), leaf_id_map)

            for left_builder, left_root in left_variants:
                right_variants = build_variants(right_sub, copy.deepcopy(left_builder), leaf_id_map)

                for right_builder, right_root in right_variants:
                    for fusion_op_class in self.fusion_operators:
                        new_builder = copy.deepcopy(right_builder)
                        fusion_op = fusion_op_class()
                        fusion_id = new_builder.create_fusion_node([left_root, right_root], fusion_op)
                        variants.append((new_builder, fusion_id))

            return variants

        n = len(leaf_infos)

        for permuted_leaf_infos in itertools.permutations(leaf_infos, n):
            base_builder = MultimodalDAGBuilder()
            leaf_id_map = {}
            for idx, (modality_id, repr_idx) in enumerate(permuted_leaf_infos):
                nodeid = base_builder.create_leaf_node(modality_id, repr_idx)
                leaf_id_map[idx] = nodeid

            indices = list(range(n))

            for tree in gen_trees(indices):
                variants = build_variants(tree, base_builder, leaf_id_map)
                for builder_variant, root_id in variants:
                    try:
                        yield builder_variant.build(root_id)
                    except ValueError:
                        if self.debug:
                            print(f"Skipping invalid DAG for root {root_id}")
                        continue

    def _evaluate_dag(self, dag: MultimodalDAG, task: Task) -> "OptimizationResult":
        start_time = time.time()
        
        try:
            fused_representation = dag.execute(self.k_best_representations, task)

            if fused_representation is None:
                return None

            final_representation = fused_representation
            if task.expected_dim == 1 and get_shape(fused_representation.metadata) > 1:
                agg_operator = AggregatedRepresentation(Aggregation())
                final_representation = agg_operator.transform(fused_representation)

            eval_start = time.time()
            scores = task.run(final_representation.data)
            eval_time = time.time() - eval_start

            total_time = time.time() - start_time

            return OptimizationResult(
                dag=dag,
                train_score=scores[0],
                val_score=scores[1],
                runtime=total_time,
                task_name=task.model.name,
                evaluation_time=eval_time,
            )

        except Exception as e:
            print(f"Error evaluating DAG: {e}")
            traceback.print_exc()
            return None

    def optimize(self, max_combinations: int = None) -> List["OptimizationResult"]:
        all_results = []

        for task in self.tasks:
            if self.debug:
                print(f"Optimizing multimodal fusion for task: {task.model.name}")

            task_results = []
            evaluated_count = 0

            for modality_subset in self._generate_modality_combinations():
                if self.debug:
                    print(f"  Evaluating modality subset: {modality_subset}")

                for repr_combo in self._generate_representation_combinations(
                    modality_subset, task.model.name
                ):

                    for dag in self._generate_fusion_dags(modality_subset, repr_combo):
                        if max_combinations and evaluated_count >= max_combinations:
                            break

                        result = self._evaluate_dag(dag, task)
                        if result is not None:
                            task_results.append(result)

                        evaluated_count += 1

                        if self.debug and evaluated_count % 100 == 0:
                            print(f"    Evaluated {evaluated_count} combinations...")

                    if max_combinations and evaluated_count >= max_combinations:
                        break

                if max_combinations and evaluated_count >= max_combinations:
                    break

            all_results.extend(task_results)

            if self.debug:
                print(
                    f"  Task completed: {len(task_results)} valid combinations evaluated"
                )

        self.optimization_results = all_results

        if self.debug:
            print(
                f"\nOptimization completed: {len(all_results)} total combinations evaluated"
            )

        return all_results


@dataclass
class OptimizationResult:
    dag: MultimodalDAG
    train_score: float
    val_score: float
    runtime: float
    task_name: str
    evaluation_time: float = 0.0
