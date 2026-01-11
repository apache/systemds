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
from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args
import json
import logging
from dataclasses import dataclass
import time
import copy

from systemds.scuro.modality.modality import Modality


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
        debug: bool = False,
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

    def get_modalities_by_id(self, modality_ids: List[int]) -> Modality:
        modalities = []
        for mod in self.modalities:
            if mod.modality_id in modality_ids:
                modalities.append(mod)
        return modalities

    def get_modality_by_id_and_instance_id(self, modality_id, instance_id):
        counter = 0
        for modality in self.modalities:
            if modality.modality_id == modality_id:
                if counter == instance_id or instance_id == -1:
                    return modality
                else:
                    counter += 1
        return None

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
                    self.optimization_results.get_k_best_results(
                        modality, self.k, task, self.scoring_metric
                    )
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
                if node.operation().parameters:
                    hyperparams.update(node.operation().parameters)
                reps.append(node.operation)
                node_order.append(node_id)
            if node.modality_id is not None:
                modality_ids.append(node.modality_id)

        visit_node(root_node_id)

        if not hyperparams:
            return None

        start_time = time.time()
        rep_name = "_".join([rep.__name__ for rep in reps])

        search_space = []
        param_names = []
        for param_name, param_values in hyperparams.items():
            param_names.append(param_name)
            if isinstance(param_values, list):
                if all(isinstance(v, (int, float)) for v in param_values):
                    if all(isinstance(v, int) for v in param_values):
                        search_space.append(
                            Integer(
                                min(param_values), max(param_values), name=param_name
                            )
                        )
                    else:
                        search_space.append(
                            Real(min(param_values), max(param_values), name=param_name)
                        )
                else:
                    search_space.append(Categorical(param_values, name=param_name))
            elif isinstance(param_values, tuple) and len(param_values) == 2:
                if isinstance(param_values[0], int) and isinstance(
                    param_values[1], int
                ):
                    search_space.append(
                        Integer(param_values[0], param_values[1], name=param_name)
                    )
                else:
                    search_space.append(
                        Real(param_values[0], param_values[1], name=param_name)
                    )
            else:
                search_space.append(Categorical([param_values], name=param_name))

        n_calls = max_evals if max_evals else 50

        all_results = []

        @use_named_args(search_space)
        def objective(**params):
            result = self.evaluate_dag_config(
                dag, params, node_order, modality_ids, task
            )
            all_results.append(result)

            score = result[1].average_scores[self.scoring_metric]
            if self.maximize_metric:
                return -score
            else:
                return score

        result = gp_minimize(
            objective,
            search_space,
            n_calls=n_calls,
            random_state=42,
            verbose=self.debug,
            n_initial_points=min(10, n_calls // 2),
        )

        if self.maximize_metric:
            best_params, best_score = max(
                all_results, key=lambda x: x[1].average_scores[self.scoring_metric]
            )
        else:
            best_params, best_score = min(
                all_results, key=lambda x: x[1].average_scores[self.scoring_metric]
            )

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

            modalities = self.get_modalities_by_id(modality_ids)
            modified_modality = dag_copy.execute(modalities, task)
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
        k: int = 1,
        optimize_unimodal: bool = True,
        max_eval_per_rep: Optional[int] = None,
    ):
        results = {}
        for task in self.tasks:
            best_results = sorted(
                optimization_results[task.model.name],
                key=lambda x: x.val_score,
                reverse=True,
            )[:k]
            results[task.model.name] = []
            best_optimization_results = best_results

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

                    result = self.tune_dag_representation(
                        dag, dag.root_node_id, task, max_eval_per_rep
                    )
                else:
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
