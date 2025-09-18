import itertools
import time
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Union, Generator
from enum import Enum
import random
import copy
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
import math
import heapq
from collections import defaultdict

from systemds.scuro.representations.aggregated_representation import (
    AggregatedRepresentation,
)
from systemds.scuro.representations.aggregate import Aggregation
from systemds.scuro.drsearch.operator_registry import Registry
from systemds.scuro.utils.schema_helpers import get_shape
from systemds.scuro.modality.transformed import TransformedModality
from systemds.scuro.modality.type import ModalityType


class SearchStrategy(Enum):
    RANDOM = "random"
    EXHAUSTIVE = "exhaustive"


@dataclass
class FusionNode:
    node_id: str
    inputs: List[str]
    operation: str
    parameters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EncoderChoice:
    modality_id: str
    modality_instance_id: str
    encoder_names: str
    encoder_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FusionArchitecture:
    encoder_choices: List[str]
    fusion_nodes: List[FusionNode]
    root_node_id: str
    used_modalities: List[str] = field(default_factory=list)

    def get_leaf_nodes(self) -> List[str]:
        return [
            f"leaf_{choice.modality_id}_{choice.modality_instance_id}"
            for choice in self.encoder_choices
        ]

    def validate(self) -> bool:
        node_ids = {node.node_id for node in self.fusion_nodes}
        leaf_ids = set(self.get_leaf_nodes())
        all_ids = node_ids | leaf_ids

        if self.root_node_id not in all_ids:
            return False

        for node in self.fusion_nodes:
            for input_id in node.inputs:
                if input_id not in all_ids:
                    return False
        return True

    def __eq__(self, other):
        if not isinstance(other, FusionArchitecture):
            return False
        return (
            self.encoder_choices == other.encoder_choices
            and self.fusion_nodes == other.fusion_nodes
            and self.root_node_id == other.root_node_id
        )

    def __hash__(self):
        encoder_tuple = tuple(
            (c.modality_id, c.modality_instance_id, c.encoder_name)
            for c in self.encoder_choices
        )
        fusion_tuple = tuple(
            (f.node_id, tuple(f.inputs), f.operation) for f in self.fusion_nodes
        )
        return hash((encoder_tuple, fusion_tuple, self.root_node_id))


class DagBuilder:
    def __init__(self, operator_registry: Registry):
        self.operator_registry = operator_registry

    def build_dag(
        self, architecture: FusionArchitecture, unimodal_representations: Dict[str, Any]
    ) -> Any:

        node_outputs = {}

        for choice in architecture.encoder_choices:
            leaf_id = f"leaf_{choice.modality_id}_{choice.modality_instance_id}"
            representation_key = f"{choice.modality_id}_{choice.modality_instance_id}"

            if representation_key in unimodal_representations:
                node_outputs[leaf_id] = unimodal_representations[representation_key]
            else:
                raise ValueError(
                    f"Missing unimodal representation: {representation_key}"
                )

        executed_nodes = set(node_outputs.keys())
        max_iterations = len(architecture.fusion_nodes) * 2
        iteration = 0

        while len(executed_nodes) < len(architecture.fusion_nodes) + len(
            architecture.encoder_choices
        ):
            if iteration > max_iterations:
                raise ValueError("Circular dependency detected in fusion architecture")

            progress_made = False

            for node in architecture.fusion_nodes:
                if node.node_id in executed_nodes:
                    continue

                if all(input_id in executed_nodes for input_id in node.inputs):
                    input_representations = [
                        node_outputs[input_id] for input_id in node.inputs
                    ]

                    fusion_ops = self.operator_registry.get_fusion_operators()
                    fusion_op = None

                    for op_class in fusion_ops:
                        if op_class.__name__ == node.operation:
                            fusion_op = op_class()
                            break

                    if fusion_op is None:
                        raise ValueError(f"Unknown fusion operation: {node.operation}")

                    if len(input_representations) == 1:
                        fused = input_representations[0]
                    else:
                        fused = input_representations[0].combine(
                            input_representations[1:], fusion_op
                        )

                    node_outputs[node.node_id] = fused
                    executed_nodes.add(node.node_id)
                    progress_made = True

            if not progress_made:
                break

            iteration += 1

        return (
            node_outputs[architecture.root_node_id]
            if architecture.root_node_id in node_outputs.keys()
            else None
        )


class SubsetModalityGenerator:
    def __init__(
        self,
        modality_encoder_choices: List[str],
        fusion_primitives: List[str],
        min_modalities: int = 1,
        max_modalities: int = None,
        max_depth: int = 4,
    ):
        self.modality_encoder_choices = modality_encoder_choices
        self.fusion_primitives = fusion_primitives
        self.min_modalities = max(1, min_modalities)
        self.max_modalities = max_modalities or len(self.modality_encoder_choices)
        self.max_depth = max_depth

    def generate_modality_subsets(self) -> Generator[List[str], None, None]:
        for r in range(
            self.min_modalities,
            min(self.max_modalities + 1, len(self.modality_encoder_choices) + 1),
        ):
            for modality_subset in itertools.permutations(
                self.modality_encoder_choices, r
            ):
                yield list(modality_subset)

    def generate_encoder_combinations_for_subset(
        self, modality_subset: List[str]
    ) -> Generator[List[EncoderChoice], None, None]:
        modality_encoder_combos = []

        for modality_id in modality_subset:
            encoder_options = self.modality_encoder_choices[modality_id]
            modality_combos = []

            if len(encoder_options) > 1:
                for r in range(1, len(encoder_options) + 1):
                    for encoder_subset in itertools.combinations(encoder_options, r):
                        encoder_choices = []
                        for i, encoder_name in enumerate(encoder_subset):
                            encoder_choices.append(
                                EncoderChoice(
                                    modality_id=modality_id,
                                    modality_instance_id=str(i),
                                    encoder_names=encoder_name,
                                )
                            )
                        modality_combos.append(encoder_choices)
            else:
                for encoder_name in encoder_options:
                    encoder_choices = [
                        EncoderChoice(
                            modality_id=modality_id,
                            modality_instance_id="0",
                            encoder_names=encoder_name,
                        )
                    ]
                    modality_combos.append(encoder_choices)

            modality_encoder_combos.append(modality_combos)

        for combo in itertools.product(*modality_encoder_combos):
            all_encoder_choices = []
            for modality_choices in combo:
                all_encoder_choices.extend(modality_choices)
            yield all_encoder_choices


class ExhaustiveFusionArchitectureGenerator(SubsetModalityGenerator):

    def generate_all_architectures(self) -> Generator[FusionArchitecture, None, None]:
        architecture_count = 0

        for modality_subset in self.generate_modality_subsets():

            leaf_nodes = [
                f"leaf_{choice.modality_id}_{choice.modality_instance_id}"
                for choice in modality_subset
            ]

            for fusion_nodes, root_node_id in self._generate_all_dags(leaf_nodes):
                architecture = FusionArchitecture(
                    encoder_choices=modality_subset,
                    fusion_nodes=fusion_nodes,
                    root_node_id=root_node_id,
                    used_modalities=modality_subset,
                )

                if architecture.validate():
                    architecture_count += 1
                    yield architecture

                    if architecture_count > 50000:
                        print(
                            f"Exhaustive search hit limit of {architecture_count} architectures"
                        )
                        return

    def _generate_all_dags(
        self, leaf_nodes: List[str]
    ) -> Generator[Tuple[List[FusionNode], str], None, None]:
        if len(leaf_nodes) == 1:
            yield [], leaf_nodes[0]
            return

        for fusion_nodes, root in self._generate_inter_modal_fusions(leaf_nodes):
            yield fusion_nodes, root

    def _generate_intra_modal_fusions(
        self, leaf_nodes: List[str]
    ) -> Generator[Dict, None, None]:
        modality_groups = defaultdict(list)
        for leaf in leaf_nodes:
            parts = leaf.split("_")
            modality_id = parts[1]
            modality_groups[modality_id].append(leaf)

        modality_fusion_options = []

        for modality_id, nodes in modality_groups.items():
            options = []

            if len(nodes) == 1:
                options.append({"remaining_nodes": nodes, "fusion_nodes": []})
            else:
                options.append({"remaining_nodes": nodes, "fusion_nodes": []})

                node_counter = 0
                for fusion_op in self.fusion_primitives:
                    fused_node_id = f"intra_{modality_id}_{node_counter}"
                    fusion_node = FusionNode(fused_node_id, nodes, fusion_op)
                    options.append(
                        {
                            "remaining_nodes": [fused_node_id],
                            "fusion_nodes": [fusion_node],
                        }
                    )
                    node_counter += 1

            modality_fusion_options.append(options)

        for combo in itertools.product(*modality_fusion_options):
            all_remaining_nodes = []
            all_fusion_nodes = []
            for option in combo:
                all_remaining_nodes.extend(option["remaining_nodes"])
                all_fusion_nodes.extend(option["fusion_nodes"])

            yield {
                "remaining_nodes": all_remaining_nodes,
                "fusion_nodes": all_fusion_nodes,
            }

    def _generate_inter_modal_fusions(
        self, nodes: List[str]
    ) -> Generator[Tuple[List[FusionNode], str], None, None]:
        if len(nodes) == 1:
            yield [], nodes[0]
            return

        if len(nodes) == 2:
            for fusion_op in self.fusion_primitives:
                fusion_node = FusionNode("fusion_0", nodes, fusion_op)
                yield [fusion_node], "fusion_0"
            return

        for combination_sequence in self._generate_combination_sequences(nodes):
            for fusion_assignment in self._assign_fusion_operations(
                combination_sequence
            ):
                yield fusion_assignment

    def _generate_combination_sequences(
        self, nodes: List[str]
    ) -> Generator[List[Tuple], None, None]:
        if len(nodes) <= 2:
            if len(nodes) == 2:
                yield [(nodes[0], nodes[1], "result_0")]
            return

        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                first_pair = (nodes[i], nodes[j], "intermediate_0")
                remaining = [n for k, n in enumerate(nodes) if k != i and k != j] + [
                    "intermediate_0"
                ]

                for rest_sequence in self._generate_combination_sequences(remaining):
                    yield [first_pair] + rest_sequence

    def _assign_fusion_operations(
        self, combination_sequence: List[Tuple]
    ) -> Generator[Tuple[List[FusionNode], str], None, None]:
        if not combination_sequence:
            return

        num_operations = len(combination_sequence)

        for fusion_ops in itertools.product(
            self.fusion_primitives, repeat=num_operations
        ):
            fusion_nodes = []

            for i, ((input1, input2, output), fusion_op) in enumerate(
                zip(combination_sequence, fusion_ops)
            ):
                fusion_node = FusionNode(output, [input1, input2], fusion_op)
                fusion_nodes.append(fusion_node)

            root_id = combination_sequence[-1][2]
            yield fusion_nodes, root_id


class RandomFusionArchitectureGenerator(SubsetModalityGenerator):

    def generate_random_architecture(self, max_depth: int = 4) -> FusionArchitecture:
        num_modalities = random.randint(
            self.min_modalities,
            min(self.max_modalities, len(self.modality_encoder_choices)),
        )
        modality_subset = random.sample(self.modality_encoder_choices, num_modalities)

        encoder_choices = []

        for modality_id in modality_subset:
            encoder_options = self.modality_encoder_choices[modality_id]

            if self.allow_intra_modal and len(encoder_options) > 1:
                num_encoders = random.randint(
                    1, min(self.max_intra_modal_per_modality, len(encoder_options))
                )
                chosen_encoders = random.sample(encoder_options, num_encoders)

                for i, encoder_name in enumerate(chosen_encoders):
                    encoder_choices.append(
                        EncoderChoice(
                            modality_id=modality_id,
                            modality_instance_id=str(i),
                            encoder_name=encoder_name,
                        )
                    )
            else:
                chosen_encoder = random.choice(encoder_options)
                encoder_choices.append(
                    EncoderChoice(
                        modality_id=modality_id,
                        modality_instance_id="0",
                        encoder_names=chosen_encoder,
                    )
                )

        fusion_nodes = []
        available_nodes = [
            f"leaf_{choice.modality_id}_{choice.modality_instance_id}"
            for choice in encoder_choices
        ]
        node_counter = 0

        if self.allow_intra_modal:
            modality_groups = {}
            for choice in encoder_choices:
                if choice.modality_id not in modality_groups:
                    modality_groups[choice.modality_id] = []
                modality_groups[choice.modality_id].append(
                    f"leaf_{choice.modality_id}_{choice.modality_instance_id}"
                )

            for modality_id, nodes in modality_groups.items():
                if len(nodes) > 1:
                    fusion_op = random.choice(self.fusion_primitives)
                    intra_modal_node_id = f"intra_{modality_id}_{node_counter}"

                    fusion_node = FusionNode(intra_modal_node_id, nodes, fusion_op)
                    fusion_nodes.append(fusion_node)

                    for node in nodes:
                        available_nodes.remove(node)
                    available_nodes.append(intra_modal_node_id)
                    node_counter += 1

        while len(available_nodes) > 1 and node_counter < max_depth:
            num_inputs = min(
                random.randint(2, min(4, len(available_nodes))), len(available_nodes)
            )
            selected_inputs = random.sample(available_nodes, num_inputs)

            fusion_op = random.choice(self.fusion_primitives)

            new_node_id = f"fusion_{node_counter}"
            fusion_node = FusionNode(new_node_id, selected_inputs, fusion_op)
            fusion_nodes.append(fusion_node)

            for node in selected_inputs:
                available_nodes.remove(node)
            available_nodes.append(new_node_id)
            node_counter += 1

        root_node_id = (
            available_nodes[0] if available_nodes else f"fusion_{node_counter-1}"
        )

        return FusionArchitecture(
            encoder_choices=encoder_choices,
            fusion_nodes=fusion_nodes,
            root_node_id=root_node_id,
            used_modalities=modality_subset,
        )


class MultimodalOptimizer:
    def __init__(
        self,
        modalities: List[Any],
        unimodal_optimization_results: Any,
        tasks: List[Any],
        k: int = 2,
        debug: bool = True,
        min_modalities: int = 1,
        max_modalities: int = None,
    ):

        self.modalities = modalities
        self.unimodal_optimization_results = unimodal_optimization_results
        self.tasks = tasks
        self.k = k
        self.debug = debug

        self.min_modalities = min_modalities
        self.max_modalities = max_modalities or len(modalities)

        self.operator_registry = Registry()
        self.fusion_primitives = [
            op.__name__ for op in self.operator_registry.get_fusion_operators()
        ]

        self.k_best_representations = self._extract_k_best_representations()
        self.modality_encoder_choices = self._create_encoder_choices()

        self.architecture_generator = RandomFusionArchitectureGenerator(
            self.modality_encoder_choices,
            self.fusion_primitives,
            min_modalities=min_modalities,
            max_modalities=self.max_modalities,
        )

        self.exhaustive_generator = ExhaustiveFusionArchitectureGenerator(
            self.modality_encoder_choices,
            self.fusion_primitives,
            min_modalities=min_modalities,
            max_modalities=self.max_modalities,
            max_depth=3,
        )

        self.dag_builder = DagBuilder(self.operator_registry)
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

    def _create_encoder_choices(self) -> List[EncoderChoice]:
        choices = []

        first_task = self.tasks[0]
        task_name = first_task.model.name

        for modality in self.modalities:
            modality_id = modality.modality_id
            k_best_data = self.k_best_representations[task_name][modality_id]
            for i, result in enumerate(k_best_data["results"]):
                r = []
                for n in result.dag.nodes:
                    if n.operation is not None:
                        r.append(n.operation.__name__)
                choices.append(
                    EncoderChoice(
                        modality_id=modality_id,
                        modality_instance_id=str(i),
                        encoder_names="".join(r),
                    )
                )

        return choices

    def _evaluate_architecture(
        self, architecture: FusionArchitecture, task: Any
    ) -> "OptimizationResult":

        start_time = time.time()
        task_name = task.model.name

        unimodal_representations = {}

        for choice in architecture.encoder_choices:
            modality_id = choice.modality_id
            encoder_names = choice.encoder_names

            task_data = self.k_best_representations[task_name][modality_id]

            selected_repr = None
            for i, result in enumerate(task_data["results"]):
                if hasattr(result, "representations"):
                    if encoder_names == f"{result.representations}":
                        selected_repr = task_data["representations"][i]
                        break

            representation_key = f"{modality_id}_{choice.modality_instance_id}"
            unimodal_representations[representation_key] = selected_repr

        fused_representation = self.dag_builder.build_dag(
            architecture, unimodal_representations
        )

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
            architecture=architecture,
            train_score=scores[0],
            val_score=scores[1],
            runtime=total_time,
            task_name=task_name,
            evaluation_time=eval_time,
        )

    def _optimize_task_exhaustive(
        self, task: Any, max_architectures: int = None
    ) -> List["OptimizationResult"]:

        task_results = []
        evaluated_count = 0

        if self.debug:
            print(f"  Starting exhaustive search for task: {task.model.name}")
            if max_architectures:
                print(f"  Limiting to first {max_architectures} architectures")

        for architecture in self.exhaustive_generator.generate_all_architectures():
            if max_architectures and evaluated_count >= max_architectures:
                break

            if self.debug and evaluated_count % 50 == 0:
                print(f"  Evaluated {evaluated_count} architectures...")

            try:
                result = self._evaluate_architecture(architecture, task)
                if result is not None:
                    task_results.append(result)
            except Exception as e:
                if self.debug:
                    print(f"  Error evaluating architecture {evaluated_count}: {e}")
                continue

            evaluated_count += 1

        if self.debug:
            print(
                f"  Exhaustive search completed: {evaluated_count} architectures evaluated"
            )
            modality_subset_counts = {}
            for result in task_results:
                num_modalities = len(result.architecture.used_modalities)
                modality_subset_counts[num_modalities] = (
                    modality_subset_counts.get(num_modalities, 0) + 1
                )
            print(f"  Modality subset distribution: {modality_subset_counts}")

        return task_results

    def optimize(
        self,
        search_strategy: SearchStrategy = SearchStrategy.RANDOM,
        search_budget: int = 50,
        **search_params,
    ) -> List["OptimizationResult"]:
        all_results = []

        for task in self.tasks:
            if self.debug:
                print(f"Optimizing fusion architectures for task: {task.model.name}")
                print(
                    f"Exploring modality subsets: {self.min_modalities} to {self.max_modalities} modalities"
                )

            task_results = self._optimize_task(
                task, search_strategy, search_budget, **search_params
            )
            all_results.extend(task_results)

        self.optimization_results = all_results

        if self.debug:
            print(
                f"\nOptimization completed: {len(all_results)} total architectures evaluated"
            )
            modality_usage = {}
            for result in all_results:
                for modality in result.architecture.used_modalities:
                    modality_usage[modality.modality_id] = (
                        modality_usage.get(modality.modality_id, 0) + 1
                    )
            print(f"Modality usage frequency: {modality_usage}")

        return all_results

    def _optimize_task(
        self,
        task: Any,
        search_strategy: SearchStrategy,
        search_budget: int,
        **search_params,
    ) -> List["OptimizationResult"]:

        if search_strategy == SearchStrategy.EXHAUSTIVE:
            max_architectures = search_params.get("max_architectures", search_budget)
            return self._optimize_task_exhaustive(task, max_architectures)

        elif search_strategy == SearchStrategy.RANDOM:
            task_results = []
            candidates = [
                self.architecture_generator.generate_random_architecture()
                for _ in range(search_budget)
            ]

            for i, architecture in enumerate(candidates):
                if self.debug and i % 10 == 0:
                    print(f"  Evaluating architecture {i+1}/{len(candidates)}")

                try:
                    result = self._evaluate_architecture(architecture, task)
                    if result is not None:
                        task_results.append(result)
                except Exception as e:
                    if self.debug:
                        print(f"  Error evaluating architecture {i}: {e}")
                    continue

            return task_results

        else:
            raise ValueError(f"Unknown search strategy: {search_strategy}")


@dataclass
class OptimizationResult:
    architecture: FusionArchitecture
    train_score: float
    val_score: float
    runtime: float
    task_name: str
    evaluation_time: float = 0.0
