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
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
from sklearn.model_selection import ParameterGrid
import json
import logging
from dataclasses import dataclass
import time
import copy

from systemds.scuro.modality.modality import Modality
from systemds.scuro.drsearch.task import Task


@dataclass
class HyperparamResult:

    representation_name: str
    best_params: Dict[str, Any]
    best_score: float
    all_results: List[Tuple[Dict[str, Any], float]]
    tuning_time: float
    modality_id: int


class HyperparameterTuner:

    def __init__(
        self,
        modalities,
        tasks,
        optimization_results,
        k: int = 2,
        n_jobs: int = -1,
        scoring_metric: str = "accuracy",
        maximize_metric: bool = True,
        save_results: bool = False,
        debug: bool = True,
    ):
        self.tasks = tasks
        self.optimization_results = optimization_results
        self.n_jobs = n_jobs
        self.scoring_metric = scoring_metric
        self.maximize_metric = maximize_metric
        self.save_results = save_results
        self.results = {}
        self.k = k
        self.modalities = modalities
        self.representations = None
        self.k_best_cache = None
        self.k_best_representations = None
        self.extract_k_best_modalities_per_task()
        self.debug = debug
        self.logger = logging.getLogger(__name__)
        if debug:
            logging.basicConfig(
                level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
            )

    def get_modality_by_id(self, modality_id: int) -> Modality:
        for mod in self.modalities:
            if mod.modality_id == modality_id:
                return mod

    def extract_k_best_modalities_per_task(self):
        self.k_best_representations = {}
        self.k_best_cache = {}
        representations = {}
        for task in self.tasks:
            self.k_best_representations[task.model.name] = []
            self.k_best_cache[task.model.name] = []
            representations[task.model.name] = {}
            for modality in self.modalities:
                k_best_results, cached_data = (
                    self.optimization_results.get_k_best_results(modality, self.k, task)
                )
                representations[task.model.name][modality.modality_id] = k_best_results
                self.k_best_representations[task.model.name].extend(k_best_results)
                self.k_best_cache[task.model.name].extend(cached_data)
        self.representations = representations

    def tune_unimodal_representations(self, max_eval_per_rep: Optional[int] = None):
        results = {}
        for task in self.tasks:
            results[task.model.name] = []
            for representation in self.k_best_representations[task.model.name]:
                result = self.tune_dag_representation(
                    representation.dag,
                    representation.dag.root_node_id,
                    task,
                    max_eval_per_rep,
                )
                results[task.model.name].append(result)

        self.results = results

        if self.save_results:
            self.save_tuning_results()

        return results

    def tune_dag_representation(self, dag, root_node_id, task, max_evals=None):
        hyperparams = {}
        reps = []
        modality_ids = []
        node_order = []

        visited = set()

        def visit_node(node_id):
            if node_id in visited:
                return
            node = dag.get_node_by_id(node_id)
            for input_id in node.inputs:
                visit_node(input_id)
            visited.add(node_id)
            if node.operation is not None:
                if node.parameters:
                    hyperparams.update(node.parameters)
                reps.append(node.operation)
                node_order.append(node_id)
            if node.modality_id is not None:
                modality_ids.append(node.modality_id)

        visit_node(root_node_id)

        if not hyperparams:
            return None

        start_time = time.time()
        rep_name = "_".join([rep.__name__ for rep in reps])

        param_grid = list(ParameterGrid(hyperparams))
        if max_evals and len(param_grid) > max_evals:
            np.random.shuffle(param_grid)
            param_grid = param_grid[:max_evals]

        all_results = []
        for params in param_grid:
            result = self.evaluate_dag_config(
                dag, params, node_order, modality_ids, task
            )
            all_results.append(result)

        if self.maximize_metric:
            best_params, best_score = max(all_results, key=lambda x: x[1])
        else:
            best_params, best_score = min(all_results, key=lambda x: x[1])

        tuning_time = time.time() - start_time

        return HyperparamResult(
            representation_name=rep_name,
            best_params=best_params,
            best_score=best_score,
            all_results=all_results,
            tuning_time=tuning_time,
            modality_id=modality_ids[0] if modality_ids else None,
        )

    def evaluate_dag_config(self, dag, params, node_order, modality_ids, task):
        try:
            dag_copy = copy.deepcopy(dag)

            for node_id in node_order:
                node = dag_copy.get_node_by_id(node_id)
                if node.operation is not None and node.parameters:
                    node_params = {
                        k: v for k, v in params.items() if k in node.parameters
                    }
                    node.parameters = node_params

            modality = self.get_modality_by_id(modality_ids[0])
            modified_modality = dag_copy.execute([modality])
            score = task.run(
                modified_modality[list(modified_modality.keys())[-1]].data
            )[1]

            return params, score
        except Exception as e:
            self.logger.error(f"Error evaluating DAG with params {params}: {e}")
            return params, float("-inf") if self.maximize_metric else float("inf")

    def tune_multimodal_representations(
        self,
        optimization_results,
        task: Task,
        k: int = 1,
        optimize_unimodal: bool = True,
        max_eval_per_rep: Optional[int] = None,
    ):
        best_optimization_results = optimization_results[:k]
        results = []
        for representation in best_optimization_results:
            if optimize_unimodal:
                dag = copy.deepcopy(representation.dag)
                index = 0
                for i, node in enumerate(representation.dag.nodes):
                    if not node.inputs:
                        leaf_node_id = node.node_id
                        leaf_nodes = self.representations[task.model.name][
                            node.modality_id
                        ][node.representation_index].dag.nodes
                        for leaf_idx, node in enumerate(dag.nodes):
                            if node.node_id == leaf_node_id:
                                dag.nodes[leaf_idx : leaf_idx + 1] = leaf_nodes
                                index = leaf_idx + len(leaf_nodes) - 1
                                break

                        for node in dag.nodes:
                            try:
                                idx = node.inputs.index(leaf_node_id)
                                node.inputs[idx] = dag.nodes[index].node_id
                                break
                            except ValueError:
                                continue

                if self._dag_has_trainable_fusion(dag):
                    result = self.tune_trainable_fusion_dag(dag, task, max_eval_per_rep)
                else:
                    result = self.tune_dag_representation(
                        dag, dag.root_node_id, task, max_eval_per_rep
                    )
            else:
                if self._dag_has_trainable_fusion(representation.dag):
                    result = self.tune_trainable_fusion_dag(
                        representation.dag, task, max_eval_per_rep
                    )
                else:
                    result = self.tune_dag_representation(
                        representation.dag,
                        representation.dag.root_node_id,
                        task,
                        max_eval_per_rep,
                    )
            results.append(result)

        self.results = results

        if self.save_results:
            self.save_tuning_results()

        return results

    def _dag_has_trainable_fusion(self, dag) -> bool:
        for node in dag.nodes:
            if node.operation and hasattr(node.operation(), "needs_training"):
                if node.operation().needs_training:
                    return True
        return False

    def tune_trainable_fusion_dag(self, dag, task, max_evals=None):
        hyperparams = {}
        reps = []
        modality_ids = []
        node_order = []
        fusion_nodes = []

        visited = set()

        def visit_node(node_id):
            if node_id in visited:
                return
            node = dag.get_node_by_id(node_id)
            for input_id in node.inputs:
                visit_node(input_id)
            visited.add(node_id)
            if node.operation is not None:
                if node.parameters:
                    hyperparams.update(node.parameters)
                reps.append(node.operation)
                node_order.append(node_id)

                if hasattr(node.operation(), "needs_training"):
                    if node.operation().needs_training:
                        fusion_nodes.append(node_id)

            if node.modality_id is not None:
                modality_ids.append(node.modality_id)

        visit_node(dag.root_node_id)

        if not hyperparams:
            return None

        start_time = time.time()
        rep_name = "_".join([rep.__name__ for rep in reps])

        param_grid = list(ParameterGrid(hyperparams))
        if max_evals and len(param_grid) > max_evals:
            np.random.shuffle(param_grid)
            param_grid = param_grid[:max_evals]

        all_results = []
        for params in param_grid:
            result = self.evaluate_trainable_fusion_config(
                dag, params, node_order, modality_ids, fusion_nodes, task
            )
            all_results.append(result)

        if self.maximize_metric:
            best_params, best_score = max(all_results, key=lambda x: x[1])
        else:
            best_params, best_score = min(all_results, key=lambda x: x[1])

        tuning_time = time.time() - start_time

        return HyperparamResult(
            representation_name=rep_name,
            best_params=best_params,
            best_score=best_score,
            all_results=all_results,
            tuning_time=tuning_time,
            modality_id=modality_ids[0] if modality_ids else None,
        )

    def evaluate_trainable_fusion_config(
        self, dag, params, node_order, modality_ids, fusion_nodes, task
    ):
        try:
            dag_copy = copy.deepcopy(dag)

            for node_id in node_order:
                node = dag_copy.get_node_by_id(node_id)
                if node.operation is not None and node.parameters:
                    node_params = {
                        k: v for k, v in params.items() if k in node.parameters
                    }
                    operation_class = node.operation
                    new_operation = operation_class(**node_params)
                    node.operation = lambda: new_operation

            required_modalities = []
            for modality_id in set(modality_ids):
                modality = self.get_modality_by_id(modality_id)
                if modality:
                    required_modalities.append(modality)

            if not required_modalities:
                raise ValueError("No valid modalities found for DAG evaluation")

            if fusion_nodes:
                final_representation = self._execute_trainable_fusion_dag(
                    dag_copy, required_modalities, task
                )
            else:
                modified_modalities = dag_copy.execute(required_modalities)
                final_representation = modified_modalities[
                    list(modified_modalities.keys())[-1]
                ]

            score = task.run(final_representation.data)[1]
            return params, score

        except Exception as e:
            self.logger.error(
                f"Error evaluating trainable fusion DAG with params {params}: {e}"
            )
            import traceback

            traceback.print_exc()
            return params, float("-inf") if self.maximize_metric else float("inf")

    def _execute_trainable_fusion_dag(self, dag, modalities, task):
        cache = {}

        def execute_node_with_training(node_id: str):
            if node_id in cache:
                return cache[node_id]

            node = dag.get_node_by_id(node_id)

            if not node.inputs:
                modality = None
                for mod in modalities:
                    if mod.modality_id == node.modality_id:
                        modality = mod
                        break
                if modality is None:
                    raise ValueError(f"Modality {node.modality_id} not found")
                cache[node_id] = modality
                return modality

            input_mods = [
                execute_node_with_training(input_id) for input_id in node.inputs
            ]

            if len(input_mods) > 1:
                fusion_op = node.operation()

                if hasattr(fusion_op, "needs_training") and fusion_op.needs_training:

                    fusion_op.transform_with_training(
                        input_mods, task.train_indices, task.labels
                    )

                    result_data = fusion_op.transform_data(input_mods, task.val_indices)

                    from systemds.scuro.modality.transformed import TransformedModality

                    result = TransformedModality(
                        modality_type="fused",
                        data=result_data,
                        metadata={"shape": result_data.shape},
                        transformation=[fusion_op],
                    )
                else:
                    result = input_mods[0].combine(input_mods[1:], fusion_op)
            else:
                if hasattr(node.operation(), "__class__"):
                    op_instance = node.operation()
                    if hasattr(input_mods[0], "apply_representation"):
                        result = input_mods[0].apply_representation(op_instance)
                    else:
                        result = op_instance.transform(input_mods[0])
                else:
                    result = input_mods[0]

            cache[node_id] = result
            return result

        final_result = execute_node_with_training(dag.root_node_id)
        return final_result

    def save_tuning_results(self, filepath: str = None):
        if not filepath:
            filepath = f"hyperparameter_results_{int(time.time())}.json"

        json_results = {}
        for task in self.results.keys():
            for result in self.results[task]:
                json_results[result.representation_name] = {
                    "best_params": result.best_params,
                    "best_score": result.best_score,
                    "tuning_time": result.tuning_time,
                    "num_evaluations": len(result.all_results),
                }

        with open(filepath, "w") as f:
            json.dump(json_results, f, indent=2)

        if self.debug:
            self.logger.info(f"Results saved to {filepath}")
